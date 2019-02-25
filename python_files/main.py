#!~/dider/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import sqlalchemy as sql
from sqlalchemy import create_engine
import numpy as np
from surprise.model_selection import train_test_split
import configparser
from timeit import default_timer
start = default_timer()
import datetime
from surprise import Dataset
from surprise import Reader
import random
from surprise.model_selection import KFold

# Import local scripts
from common import common
from dataIntegration import dataStandardization
from trainAlgorithm import trainAlgorithm 
from hybrid import hybrid
import evaluation 
import os
from gridSearch import gridSearch

class main:

    def __init__(self, client_name):
        # Nombre del cliente para identificar sobre cuales archivos se ejecutaran los procesos
        self.client_name = client_name+'/'

        # Archivo de configuracion
        config = configparser.ConfigParser()
        config.sections()
        #config.read('config.ini', encoding="utf8")
        if os.path.isfile(str('../Datasets/'+self.client_name)+'config.ini'):
            with open(str('../Datasets/'+self.client_name)+'config.ini') as config_parser_fp:
                config.read_file(config_parser_fp)
            
        # Variables de configurción
        self.verbose_switch = config['DEFAULT'].getboolean('verbose_switch')
        self.min_interactions_explicit = int(config['CLEANING']['min_interactions_explicit'])
        self.recommendations_per_user = int(config['RECOMMEND']['recommendations_per_user'])
        self.number_of_folds = int(config['SAMPLING']['number_of_folds'])
        self.test_size_conf = float(config['SAMPLING']['test_size'])
        self.maximum_interactions_evaluation = int(config['SAMPLING']['maximum_interactions_evaluation'])
        self.statistical_significance = config['RESULTS'].getboolean('statistical_significance')
        self.number_of_2_fold_samples = int(config['RESULTS']['number_of_2_fold_samples'])
        self.min_population_constraint = int(config['RESULTS']['min_population_constraint'])

        self.data_path = "../Datasets/"+str(self.client_name)+"data/"
        self.data_path_backup = "../Datasets/"+str(self.client_name)+"data_backup/"
        self.database_path = "../Datasets/"+str(self.client_name)+"database/input_data/"
        self.models_path = "../Datasets/"+str(self.client_name)+"database/models/"
        self.valid_data_directories = ["implicit","explicit","explicit_review","user_content","item_content"]
        self.valid_data_conf_names = ["implicit_conf.json","explicit_conf.json","explicit_review_conf.json","user_content_conf.json","item_content_conf.json"]
        self.sql_db = sql.create_engine('sqlite:///'+self.database_path+"db.sql", encoding='utf-8')
        self.sql_db.raw_connection().connection.text_factory = str

        self.surprise_models=["surprise_svd","surprise_SVDpp","surprise_NMF","surprise_NormalPredictor","surprise_BaselineOnly","surprise_KNNBasic", "surprise_KNNWithMeans", "surprise_KNNWithZScore", "surprise_KNNBaseline", "surprise_SlopeOne", "surprise_CoClustering"]
        self.explicit_models_to_run = ["surprise_svd","surprise_SVDpp","surprise_NMF","surprise_NormalPredictor","surprise_BaselineOnly","surprise_KNNBasic", "surprise_KNNWithMeans", "surprise_KNNWithZScore", "surprise_KNNBaseline", "surprise_SlopeOne", "surprise_CoClustering"]
        self.explicit_review_models_to_run = ["surprise_svd","surprise_SVDpp","surprise_NMF","surprise_NormalPredictor","surprise_BaselineOnly","surprise_KNNBasic", "surprise_KNNWithMeans", "surprise_KNNWithZScore", "surprise_KNNBaseline", "surprise_SlopeOne", "surprise_CoClustering"]
        self.common_functions = common(self.client_name)
        self.training = trainAlgorithm(self.client_name)
        self.gridSearch = gridSearch(self.client_name)
        self.hybrid = hybrid(self.client_name)

    # Validar directorios definidos en configuración
    def initialize_environment(self):
        
        init_result = self.common_functions.initialize(self.data_path,self.data_path_backup,self.database_path,self.valid_data_directories)
        print(init_result)

    def get_source_data(self):
        self.initialize_environment()
        print("------------------------------------------------")
        print("STARTING DATA SOURCES EXTRACTION")
        print("------------------------------------------------")
        st = default_timer()
        dataIntegration = dataStandardization(self.client_name)
        source_result = dataIntegration.data_etl(self.valid_data_conf_names, self.data_path, self.data_path_backup, self.sql_db, self.database_path)
        runtime = default_timer() - st
        print ("Elapsed time(sec): ", round(runtime,2))
        if(source_result["status"] == True):
            print(source_result["result"])
        else:
            print("There was an error trying to process data sources in ETL process: "+source_result["result"])

        # Almacenar tiempo de proceso en base de datos
        new_entry= pd.DataFrame(columns=['event','description','process_time','datetime'])
        new_entry = new_entry.append({'event':"get_knowledge_sources",'description': "Tiempo total en transformar y extraer fuentes de conocimiento","process_time":round(runtime,2), "datetime":datetime.datetime.now()}, ignore_index=True)
        new_entry.to_sql("time_results", self.sql_db, if_exists='append')
        del(new_entry)
        print("------------------------------------------------")
        print("ENDING DATA SOURCES EXTRACTION")
        print("------------------------------------------------")


    def run_and_evaluate_algorithms(self, k=None, min_interactions_per_user = None, num_recommend = None, new_run = True, explicit_models = None, explicit_review_models=None, new=True, statistical_significance=None):
        if k is None:
            k=self.number_of_folds
        if min_interactions_per_user is None:
            min_interactions_per_user=self.min_interactions_explicit
        if num_recommend is None:
            num_recommend=self.recommendations_per_user
        if explicit_models is None:
            explicit_models=self.explicit_models_to_run
        if explicit_review_models is None:
            explicit_review_models=self.explicit_review_models_to_run
        if statistical_significance is None:
            statistical_significance=self.statistical_significance

        st = default_timer()
        print("------------------------------------------------")
        print("STARTING ALGORITHM TRAINING AND EVALUATION")
        print("------------------------------------------------")
        
        print("K-fold: "+str(k))
        print("Filter - min interactions per user: "+str(min_interactions_per_user))
        print("Number of recommendations per user: "+str(num_recommend))
        print("Explicit models to evaluate: "+str(explicit_models))
        print("Explicit models to evaluate using text reviews: "+str(explicit_review_models))

        # Limpiar las tablas de resultados
        #self.common_functions.reset_result_tables(sql_db)

        # Obtener el contenido de los items (si existe)
        item_content, item_weights = self.common_functions.get_item_content(self.sql_db, self.database_path)

        # Obtener el contenido de los usuarios (si existe)
        user_content = self.common_functions.get_user_content(self.sql_db)

        available_data = self.common_functions.get_available_data(self.sql_db)
        

        # Explicit
        if("explicit" in available_data):
            print("Evaluating and selecting for Explicit data .......")
            st_explicit = default_timer()
            print("Removing users without enought interactions in explicit dataset")
            # Cargar en un dataframe los datos explicitos
            explicit = pd.read_sql_query('select * from explicit;', self.sql_db, index_col='index')
            # Para cada tipo de datos en available_data, hacer transformaciónes según parametros de entrada
            # sacar usuarios que no hayan interactuado con el numero minimo de items definido en los parametros de entrada
            if(min_interactions_per_user > 0):
                users_interactions_count_df = explicit.groupby(['user_id', 'item_id']).size().groupby('user_id').size()
                print('# users: %d' % len(users_interactions_count_df))
                users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= min_interactions_per_user].reset_index()[['user_id']]
                print('# users with at least '+str(min_interactions_per_user)+' interactions: %d' % len(users_with_enough_interactions_df))

                print('# of interactions: %d' % len(explicit))
                interactions_from_selected_users_df = explicit.merge(users_with_enough_interactions_df, 
                            how = 'right',
                            left_on = 'user_id',
                            right_on = 'user_id')
                print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
                del(users_with_enough_interactions_df)
                del(users_interactions_count_df)
                input_data = interactions_from_selected_users_df
            else:
                input_data = explicit
            del(explicit)
            print("Removing users DONE")

            dataset_len = len(input_data.index)
            if((self.min_population_constraint == 0) or (dataset_len <= self.min_population_constraint)):
                statistical_significance = False

            # Si no se requiere test de significancia estadistica
            if(statistical_significance == False):
                # Split dataset for evaluation
                #dataset_len = len(input_data.index)
                if(dataset_len > self.maximum_interactions_evaluation):
                    input_data = input_data.sample(n=maximum_interactions_evaluation, random_state=101).reset_index(drop=True)
                
                if(len(self.surprise_models) != 0):
                    
                    # Transformar datos para ser usados por los modelos de la librerira Surprise
                    # Extraer del dataframe solo las columnas que usa la libreria Surprise
                    train_data = input_data[["item_id","user_id","target"]]
                    # Cambiarlos nombres de las columnas
                    train_data=train_data.rename(columns = {"item_id":"itemID","user_id":"userID","target":"rating"})
                    #Obtener los valores maximos y minimos para la columna rating del DataFrame
                    max_rating = train_data["rating"].max()
                    min_rating = train_data["rating"].min()
                    # Se crea el objeto reader requerido por la libreria para usar el DataFrame
                    reader = Reader(rating_scale=(min_rating, max_rating))
                    # Se lee el DataFrame con la libreria Surprise
                    data = Dataset.load_from_df(train_data[['userID', 'itemID', 'rating']], reader)
                    
                    print("STARTING GridSearch to find best parameters for algorithms")
                    self.gridSearch.parameters_selection(data, self.surprise_models, "explicit", new=new)
                    print("GridSearch DONE")
                    
                    if(self.number_of_folds == 1):
                        kfold = 1
                        trainset, testset = train_test_split(data, test_size=test_size_conf)
                        print("Starting kfold "+str(kfold))
                        self.training.characterize_algorithms("explicit", trainset, testset, sql_db, kfold, k_recommend = num_recommend, models_to_run=surprise_models, system_eval=False)
                        print("kfold "+str(kfold)+" DONE")
                    elif(self.number_of_folds > 1):
                        kfold = 1
                        kf = KFold(n_splits=self.number_of_folds)
                        for trainset, testset in kf.split(data):
                            print("Starting kfold "+str(kfold))
                            self.training.characterize_algorithms("explicit", trainset, testset, self.sql_db, kfold, k_recommend = num_recommend, models_to_run=self.surprise_models, system_eval=False)
                            print("kfold "+str(kfold)+" DONE")
                            kfold += 1
            else:

                for n in range(1, self.number_of_2_fold_samples+1):
                    # Split dataset for evaluation
                    #dataset_len = len(input_data.index)
                    if(dataset_len > self.maximum_interactions_evaluation):
                        input_data = input_data.sample(n=maximum_interactions_evaluation).reset_index(drop=True)
                    
                    if(len(self.surprise_models) != 0):
                        
                        # Transformar datos para ser usados por los modelos de la librerira Surprise
                        # Extraer del dataframe solo las columnas que usa la libreria Surprise
                        train_data = input_data[["item_id","user_id","target"]]
                        # Cambiarlos nombres de las columnas
                        train_data=train_data.rename(columns = {"item_id":"itemID","user_id":"userID","target":"rating"})
                        #Obtener los valores maximos y minimos para la columna rating del DataFrame
                        max_rating = train_data["rating"].max()
                        min_rating = train_data["rating"].min()
                        # Se crea el objeto reader requerido por la libreria para usar el DataFrame
                        reader = Reader(rating_scale=(min_rating, max_rating))
                        # Se lee el DataFrame con la libreria Surprise
                        data = Dataset.load_from_df(train_data[['userID', 'itemID', 'rating']], reader)
                        
                        print("STARTING GridSearch to find best parameters for algorithms")
                        self.gridSearch.parameters_selection(data, self.surprise_models, "explicit", new=new)
                        print("GridSearch DONE")

                        
                        kfold = n
                        trainset, testset = train_test_split(data, test_size=self.test_size_conf)
                        print("Starting kfold "+str(kfold))
                        self.training.characterize_algorithms("explicit", trainset, testset, self.sql_db, kfold, k_recommend = num_recommend, models_to_run=self.surprise_models, system_eval=False)
                        print("kfold "+str(kfold)+" DONE")
                        
            runtime_explicit = default_timer() - st_explicit
            print("DONE Evaluating and selecting for Explicit data. Time (Seconds): "+str(round(runtime_explicit,2)))

            # Presentar los resultados de evaluación para los algoritmos ejecutados
            self.hybrid.select_best_results(knowledge_name="explicit", hybrid_list=False)

            # Almacenar tiempo de proceso en base de datos
            new_entry= pd.DataFrame(columns=['event','description','process_time','datetime'])
            new_entry = new_entry.append({'event':"explicit_evaluation_selection",'description': "Tiempo total en evaluar y seleccionar los algoritmos: "+str(explicit_models)+" para los datos explicitos disponibles. K-fold: "+str(k),"process_time":round(runtime_explicit,2), "datetime":datetime.datetime.now()}, ignore_index=True)
            new_entry.to_sql("time_results", self.sql_db, if_exists='append')
            del(new_entry)
     
        runtime_eval_select = default_timer() - st

        # Almacenar tiempo de proceso en base de datos
        new_entry= pd.DataFrame(columns=['event','description','process_time','datetime'])
        new_entry = new_entry.append({'event':"algorithms_evaluation_selection",'description': "Tiempo en evaluar y seleccionar los algoritmos optimos para las fuentes de datos disponibles. Fuentes disponibles: "+str(available_data)+". K-fold: "+str(k),"process_time":round(runtime_eval_select,2), "datetime":datetime.datetime.now()}, ignore_index=True)
        new_entry.to_sql("time_results", self.sql_db, if_exists='append')
        del(new_entry)

        print("------------------------------------------------")
        print("ENDING ALGORITHM TRAINING AND EVALUATION")
        print("------------------------------------------------")

        print("------------------------------------------------")
        print("STARTING TO HYBRIDATE ALGORITHMS")
        print("------------------------------------------------")
        self.hybridate_best_results(kfold=self.number_of_folds, k_recommend=self.recommendations_per_user, min_interactions_per_user=self.min_interactions_explicit, statistical_significance=self.statistical_significance)
        print("------------------------------------------------")
        print("ALGORITHM HYBRIDATION IS DONE")
        print("------------------------------------------------")
        
        self.hybrid.select_best_results(knowledge_name="explicit", hybrid_list=False)

        print("------------------------------------------------")
        print("ALGORITHM WITH BEST RESULTS")
        print("------------------------------------------------")
        self.hybrid.select_best_algorithm_knowledge("explicit")

        if(self.statistical_significance == True):
            print("------------------------------------------------")
            print("STATISTICAL SIGNIFICANCE")
            print("------------------------------------------------")
            knowledge_result = pd.read_sql_query('select * from explicit_totals;', self.sql_db, index_col='index')
            knowledge_result = knowledge_result.sort_values("total",ascending=False)
            print("Positions 1 and 2")
            self.hybrid.get_significance(knowledge_result["model"].iloc[0], knowledge_result["model"].iloc[1], "total")
            print("Positions 2 and 3")
            self.hybrid.get_significance(knowledge_result["model"].iloc[1], knowledge_result["model"].iloc[2], "total")

    def hybridate_explicit_best(self, kfold, k_recommend):
        import itertools as it
        # SELECCIONAR MEJORES ALGORITMOS PARA EXPLICIT POR CADA TECNICA
        hybridation_results = self.hybrid.select_best_results(knowledge_name="explicit", hybrid_list=True)
        combined_algorithms = list(it.combinations(hybridation_results, 2))
        print("Algorithms to hybridate: "+str(hybridation_results))

        for pair in combined_algorithms:
            print(pair[0])
            self.hybrid.hybrid_algorithms(pair[0], pair[1], kfold, "explicit", k_recommend)

    def hybridate_best_results(self, kfold, k_recommend, min_interactions_per_user, statistical_significance):
        available_data = self.common_functions.get_available_data(self.sql_db)
        if("explicit" in available_data):
            explicit = pd.read_sql_query('select * from explicit;', self.sql_db, index_col='index')
            # Para cada tipo de datos en available_data, hacer transformaciónes según parametros de entrada
            # sacar usuarios que no hayan interactuado con el numero minimo de items definido en los parametros de entrada
            if(min_interactions_per_user > 0):
                users_interactions_count_df = explicit.groupby(['user_id', 'item_id']).size().groupby('user_id').size()
                print('# users: %d' % len(users_interactions_count_df))
                users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= min_interactions_per_user].reset_index()[['user_id']]
                print('# users with at least '+str(min_interactions_per_user)+' interactions: %d' % len(users_with_enough_interactions_df))

                print('# of interactions: %d' % len(explicit))
                interactions_from_selected_users_df = explicit.merge(users_with_enough_interactions_df, 
                            how = 'right',
                            left_on = 'user_id',
                            right_on = 'user_id')
                print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
                del(users_with_enough_interactions_df)
                del(users_interactions_count_df)
                input_data = interactions_from_selected_users_df
            else:
                input_data = explicit
            del(explicit)
            dataset_len = len(input_data.index)
            del(input_data)
            if((self.min_population_constraint == 0) or (dataset_len <= self.min_population_constraint)):
                statistical_significance = False

            # Si no se requiere test de significancia estadistica
            if(statistical_significance == False):
                self.hybridate_explicit_best(kfold=kfold, k_recommend=k_recommend)
            else:
                self.hybridate_explicit_best(kfold=self.number_of_2_fold_samples, k_recommend=k_recommend)
