#!~/dider/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division

# Import local scripts
from evaluation import evaluation
import time
import configparser
import pandas as pd
import os
from surprise import SVD
from surprise import SlopeOne
from surprise import CoClustering
from surprise import SVDpp
from surprise import NMF
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import Reader
from surprise import dump
from timeit import default_timer
start = default_timer()
import sys
import json
from common import common
import recommend

class trainAlgorithm:

    def __init__(self, client_name):
        # Nombre del cliente para identificar sobre cuales archivos se ejecutaran los procesos
        self.client_name = client_name+'/'
        # Archivo de configuracion
        self.config = configparser.ConfigParser()
        self.config.sections()
        #config.read('config.ini', encoding="utf8")
        if os.path.isfile(str('../Datasets/'+self.client_name)+'config.ini'):
            with open(str('../Datasets/'+self.client_name)+'config.ini') as config_parser_fp:
                self.config.read_file(config_parser_fp)
        # Configuracion general
        self.models_path = "../Datasets/"+str(self.client_name)+"database/models/"
        self.database_path = "../Datasets/"+str(self.client_name)+"database/input_data/"
        self.common_functions = common(self.client_name)
        self.evaluation = evaluation(self.client_name)

    def characterize_algorithms(self, knowledge_type, train_data, test_data, sql_db, kfold, k_recommend = 10, models_to_run=["surprise_matrix_svd"], system_eval=False):
        executed_algorithms = []
        if("surprise_svd" in models_to_run):
            print("Comienza surprise_svd_"+str(knowledge_type))
            exec_result = self.train_surprise("svd", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_svd_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_svd_"+str(knowledge_type))
                print("Termina surprise_svd_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_SVDpp" in models_to_run):
            print("Comienza surprise_SVDpp_"+str(knowledge_type))
            exec_result = self.train_surprise("SVDpp", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_SVDpp_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_SVDpp_"+str(knowledge_type))
                print("Termina surprise_SVDpp_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_NMF" in models_to_run):
            print("Comienza surprise_NMF_"+str(knowledge_type))
            exec_result = self.train_surprise("NMF", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_NMF_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_NMF_"+str(knowledge_type))
                print("Termina surprise_NMF_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_NormalPredictor" in models_to_run):
            print("Comienza surprise_NormalPredictor_"+str(knowledge_type))
            exec_result = self.train_surprise("NormalPredictor", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_NormalPredictor_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_NormalPredictor_"+str(knowledge_type))
                print("Termina surprise_NormalPredictor_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_BaselineOnly" in models_to_run):
            print("Comienza surprise_BaselineOnly_"+str(knowledge_type))
            exec_result = self.train_surprise("BaselineOnly", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_BaselineOnly_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_BaselineOnly_"+str(knowledge_type))
                print("Termina surprise_BaselineOnly_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_KNNBasic" in models_to_run):
            print("Comienza surprise_KNNBasic_"+str(knowledge_type))
            exec_result = self.train_surprise("KNNBasic", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_KNNBasic_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_KNNBasic_"+str(knowledge_type))
                print("Termina surprise_KNNBasic_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_KNNWithMeans" in models_to_run):
            print("Comienza surprise_KNNWithMeans_"+str(knowledge_type))
            exec_result = self.train_surprise("KNNWithMeans", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_KNNWithMeans_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_KNNWithMeans_"+str(knowledge_type))
                print("Termina surprise_KNNWithMeans_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_KNNWithZScore" in models_to_run):
            print("Comienza surprise_KNNWithZScore_"+str(knowledge_type))
            exec_result = self.train_surprise("KNNWithZScore", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_KNNWithZScore_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_KNNWithZScore_"+str(knowledge_type))
                print("Termina surprise_KNNWithZScore_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_KNNBaseline" in models_to_run):
            print("Comienza surprise_KNNBaseline_"+str(knowledge_type))
            exec_result = self.train_surprise("KNNBaseline", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_KNNBaseline_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_KNNBaseline_"+str(knowledge_type))
                print("Termina surprise_KNNBaseline_"+str(knowledge_type))
            else:
                print(exec_result["result"])


        if("surprise_SlopeOne" in models_to_run):
            print("Comienza surprise_SlopeOne_"+str(knowledge_type))
            exec_result = self.train_surprise("SlopeOne", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_SlopeOne_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_SlopeOne_"+str(knowledge_type))
                print("Termina surprise_SlopeOne_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        if("surprise_CoClustering" in models_to_run):
            print("Comienza surprise_CoClustering_"+str(knowledge_type))
            exec_result = self.train_surprise("CoClustering", train_data, test_data,  k_recommend, sql_db, kfold, knowledge=knowledge_type, model_name="surprise_CoClustering_"+str(knowledge_type), result_name="collaborative_results", system_eval=system_eval)
            if(exec_result["status"] == True):
                executed_algorithms.append("surprise_CoClustering_"+str(knowledge_type))
                print("Termina surprise_CoClustering_"+str(knowledge_type))
            else:
                print(exec_result["result"])

        # ------------------------------------
    def train_surprise(self, model_type, trainset, testset, k_recommend, sql_db, k_fold, knowledge, model_name, result_name, system_eval=False):

        knn_user_based = self.config['SURPRISE_KNN'].getboolean('knn_user_based')
        knn_similarity = self.config['SURPRISE_KNN']['knn_similarity']
        sim_options = {'name': knn_similarity,'user_based': knn_user_based}
        verbose_switch = self.config['DEFAULT'].getboolean('verbose_switch')
        # Selección de modelo a utilizar
        if(model_type == "svd"):
            # Obtener valores de configuracion
            svd_grid_search = self.config['SURPRISE_SVD'].getboolean('svd_grid_search')
            svd_grid_metric = self.config['SURPRISE_SVD']['svd_grid_metric']
            svd_n_factors = int(self.config['SURPRISE_SVD']['svd_n_factors'])
            svd_n_epochs = int(self.config['SURPRISE_SVD']['svd_n_epochs'])
            svd_biased = self.config['SURPRISE_SVD'].getboolean('svd_biased')
            svd_init_mean = float(self.config['SURPRISE_SVD']['svd_init_mean'])
            svd_init_std_dev = float(self.config['SURPRISE_SVD']['svd_init_std_dev'])
            svd_lr_all = float(self.config['SURPRISE_SVD']['svd_lr_all'])
            svd_reg_all = float(self.config['SURPRISE_SVD']['svd_reg_all'])

            if(self.common_functions.validate_available_sql_data('svd_params',sql_db) == True):
                results = pd.read_sql_query('select * from svd_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="svd")]
                if(real_results.empty == False):
                    svd_n_factors = int(real_results.iloc[0]['svd_n_factors'])
                    svd_n_epochs = int(real_results.iloc[0]['svd_n_epochs'])
                    svd_init_std_dev = float(real_results.iloc[0]['svd_init_std_dev'])
                    svd_lr_all = float(real_results.iloc[0]['svd_lr_all'])
                    svd_reg_all = float(real_results.iloc[0]['svd_reg_all'])

            algo = SVD(n_factors=svd_n_factors, n_epochs=svd_n_epochs, biased=svd_biased, init_mean=svd_init_mean, init_std_dev=svd_init_std_dev, lr_all=svd_lr_all, reg_all=svd_reg_all, verbose=verbose_switch)
        
        elif(model_type == "SVDpp"):
            # Obtener valores de configuracion
            svdpp_grid_search = self.config['SURPRISE_SVDPP'].getboolean('svdpp_grid_search')
            svdpp_grid_metric = self.config['SURPRISE_SVDPP']['svdpp_grid_metric']
            svdpp_n_factors = int(self.config['SURPRISE_SVDPP']['svdpp_n_factors'])
            svdpp_n_epochs = int(self.config['SURPRISE_SVDPP']['svdpp_n_epochs'])
            svdpp_init_mean = float(self.config['SURPRISE_SVDPP']['svdpp_init_mean'])
            svdpp_init_std_dev = float(self.config['SURPRISE_SVDPP']['svdpp_init_std_dev'])
            svdpp_lr_all = float(self.config['SURPRISE_SVDPP']['svdpp_lr_all'])
            svdpp_reg_all = float(self.config['SURPRISE_SVDPP']['svdpp_reg_all'])

            if(self.common_functions.validate_available_sql_data('svdpp_params',sql_db) == True):
                results = pd.read_sql_query('select * from svdpp_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="svdpp")]
                if(real_results.empty == False):
                    svdpp_n_factors = int(real_results.iloc[0]['svdpp_n_factors'])
                    svdpp_n_epochs = int(real_results.iloc[0]['svdpp_n_epochs'])
                    svdpp_init_std_dev = float(real_results.iloc[0]['svdpp_init_std_dev'])
                    svdpp_lr_all = float(real_results.iloc[0]['svdpp_lr_all'])
                    svdpp_reg_all = float(real_results.iloc[0]['svdpp_reg_all'])

            algo = SVDpp(n_factors=svdpp_n_factors, n_epochs=svdpp_n_epochs, init_mean=svdpp_init_mean, init_std_dev=svdpp_init_std_dev, lr_all=svdpp_lr_all, reg_all=svdpp_reg_all, verbose=verbose_switch)
        
        elif(model_type == "NMF"):
            # Obtener valores de configuracion
            nmf_grid_search = self.config['SURPRISE_NMF'].getboolean('nmf_grid_search')
            nmf_grid_metric = self.config['SURPRISE_NMF']['nmf_grid_metric']
            nmf_n_factors = int(self.config['SURPRISE_NMF']['nmf_n_factors'])
            nmf_n_epochs = int(self.config['SURPRISE_NMF']['nmf_n_epochs'])
            nmf_biased = self.config['SURPRISE_NMF'].getboolean('nmf_biased')
            nmf_reg_pu = float(self.config['SURPRISE_NMF']['nmf_reg_pu'])
            nmf_reg_qi = float(self.config['SURPRISE_NMF']['nmf_reg_qi'])
            nmf_reg_bu = float(self.config['SURPRISE_NMF']['nmf_reg_bu'])
            nmf_reg_bi = float(self.config['SURPRISE_NMF']['nmf_reg_bi'])
            nmf_lr_bu = float(self.config['SURPRISE_NMF']['nmf_lr_bu'])
            nmf_lr_bi = float(self.config['SURPRISE_NMF']['nmf_lr_bi'])
            nmf_init_low = float(self.config['SURPRISE_NMF']['nmf_init_low'])
            nmf_init_high = int(self.config['SURPRISE_NMF']['nmf_init_high'])

            if(self.common_functions.validate_available_sql_data('nmf_params',sql_db) == True):
                results = pd.read_sql_query('select * from nmf_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="nmf")]
                if(real_results.empty == False):
                    nmf_n_factors = int(real_results.iloc[0]['nmf_n_factors'])
                    nmf_n_epochs = int(real_results.iloc[0]['nmf_n_epochs'])
                    nmf_reg_pu = float(real_results.iloc[0]['nmf_reg_pu'])
                    nmf_reg_qi = float(real_results.iloc[0]['nmf_reg_qi'])
                    nmf_init_low = float(real_results.iloc[0]['nmf_init_low'])

            algo = NMF(n_factors=nmf_n_factors, n_epochs=nmf_n_epochs, biased=nmf_biased, reg_pu=nmf_reg_pu, reg_qi=nmf_reg_qi, reg_bu=nmf_reg_bu, reg_bi=nmf_reg_bi, lr_bu=nmf_lr_bu, lr_bi=nmf_lr_bi, init_low=nmf_init_low, init_high=nmf_init_high, verbose=verbose_switch)
        
        elif(model_type == "NormalPredictor"):
            algo = NormalPredictor()
        
        elif(model_type == "BaselineOnly"):
            algo = BaselineOnly(verbose=verbose_switch)
        
        elif(model_type == "KNNBasic"):
            # Obtener valores de configuracion
            knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
            knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
            knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
            knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

            if(self.common_functions.validate_available_sql_data('knnbasic_params',sql_db) == True):
                results = pd.read_sql_query('select * from knnbasic_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="knnbasic")]
                if(real_results.empty == False):
                    knn_k = int(real_results.iloc[0]['knn_k'])
                    knn_min_k = int(real_results.iloc[0]['knn_min_k'])
            
            algo = KNNBasic(k=knn_k, min_k=knn_min_k, sim_options=sim_options, verbose=verbose_switch)
        
        elif(model_type == "KNNWithMeans"):
            # Obtener valores de configuracion
            knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
            knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
            knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
            knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

            if(self.common_functions.validate_available_sql_data('knnwithmeans_params',sql_db) == True):
                results = pd.read_sql_query('select * from knnwithmeans_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="knnwithmeans")]
                if(real_results.empty == False):
                    knn_k = int(real_results.iloc[0]['knn_k'])
                    knn_min_k = int(real_results.iloc[0]['knn_min_k'])

            algo = KNNWithMeans(k=knn_k, min_k=knn_min_k, sim_options=sim_options, verbose=verbose_switch)
        
        elif(model_type == "KNNWithZScore"):
            # Obtener valores de configuracion
            knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
            knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
            knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
            knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

            if(self.common_functions.validate_available_sql_data('knnwithzscore_params',sql_db) == True):
                results = pd.read_sql_query('select * from knnwithzscore_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="knnwithzscore")]
                if(real_results.empty == False):
                    knn_k = int(real_results.iloc[0]['knn_k'])
                    knn_min_k = int(real_results.iloc[0]['knn_min_k'])

            algo = KNNWithZScore(k=knn_k, min_k=knn_min_k, sim_options=sim_options, verbose=verbose_switch)
        
        elif(model_type == "KNNBaseline"):
            # Obtener valores de configuracion
            knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
            knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
            knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
            knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

            if(self.common_functions.validate_available_sql_data('knnbaseline_params',sql_db) == True):
                results = pd.read_sql_query('select * from knnbaseline_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="knnbaseline")]
                if(real_results.empty == False):
                    knn_k = int(real_results.iloc[0]['knn_k'])
                    knn_min_k = int(real_results.iloc[0]['knn_min_k'])

            algo = KNNBaseline(k=knn_k, min_k=knn_min_k, sim_options=sim_options, verbose=verbose_switch)
        
        elif(model_type == "SlopeOne"):
            algo = SlopeOne()
        
        elif(model_type == "CoClustering"):
            # Obtener valores de configuracion
            cc_grid_search = self.config['SURPRISE_COCLUSTERING'].getboolean('cc_grid_search')
            cc_grid_metric = self.config['SURPRISE_COCLUSTERING']['cc_grid_metric']
            cc_n_cltr_u = int(self.config['SURPRISE_COCLUSTERING']['cc_n_cltr_u'])
            cc_n_cltr_i = int(self.config['SURPRISE_COCLUSTERING']['cc_n_cltr_i'])
            cc_n_epochs = int(self.config['SURPRISE_COCLUSTERING']['cc_n_epochs'])
            
            if(self.common_functions.validate_available_sql_data('coclustering_params',sql_db) == True):
                results = pd.read_sql_query('select * from coclustering_params;', sql_db, index_col='index')
                real_results = results[(results["knowledge"]==knowledge) & (results["algorithm"]=="coclustering")]
                if(real_results.empty == False):
                    cc_n_cltr_u = int(real_results.iloc[0]['cc_n_cltr_u'])
                    cc_n_cltr_i = int(real_results.iloc[0]['cc_n_cltr_i'])
                    cc_n_epochs = int(real_results.iloc[0]['cc_n_epochs'])

            algo = CoClustering(n_cltr_u=cc_n_cltr_u, n_cltr_i=cc_n_cltr_i, n_epochs=cc_n_epochs, verbose=verbose_switch)
        else:
            return {"status":False, "result":"Defined model_type does not exist"}

        st = default_timer()
        print("STARTING to train model: "+str(model_name))
        algo.fit(trainset)
        train_model_runtime = default_timer() - st
        # Almacenar tiempo de proceso en base de datos
        self.common_functions.save_process_time(st, event = str(model_name)+"_training", description = "Time for model to be trained on dataset")

        # Guardar modelo
        # Crear directorio si no existe
        if(os.path.isdir(self.models_path+model_name) == False):
            try:
                os.makedirs(self.models_path+model_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    return {"status": False, "result":e}
        # Almacenar modelo en file system
        #file_name =  self.models_path+model_name+"/model"
        #dump.dump(file_name, algo=algo)

        st = default_timer()
        print("STARTING to generate predictions with the trained model: "+str(model_name))
        predictions = algo.test(testset)
        runtime = default_timer() - st

        print ("Tiempo de ejecucion total de la generacion de predicciones para Surprise Time:", round(runtime,2))
        self.common_functions.save_process_time(st, event = str(model_name)+"_generate_recommendations", description = "Time for predictions to be generated using the model")

        # Guardar predicciones para hibridación
        # Crear directorio si no existe
        if(os.path.isdir(self.models_path+model_name+"/predictions/"+str(k_fold)) == False):
            try:
                os.makedirs(self.models_path+model_name+"/predictions/"+str(k_fold))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    return {"status": False, "result":e}

        # Almacenar predicciones para hibridación
        eval_result= pd.DataFrame(columns=['user_id','item_id','r_ui','est'])
        for uid, iid, true_r, est, _ in predictions:
            eval_result=eval_result.append({'user_id': uid,'item_id': iid,'r_ui': true_r,'est': est}, ignore_index=True)
        eval_result.to_csv(path_or_buf=self.models_path+model_name+"/predictions/"+str(k_fold)+"/predictions.csv", encoding='latin1', sep=str(u';').encode('utf-8'), index=False)

        
        # ---------------------------

        if(system_eval==False):
            # Procesar y evaluar las recomendaciones para el modelo
            st = default_timer()
            print("STARTING to evaluate recommendations with model: "+str(model_name))
            process_evaluate_result = self.evaluation.surprise_process_evaluate(predictions, knowledge, model_name, result_name, train_model_runtime, k_recommend, sql_db, k_fold, is_surprise=True)
            # Almacenar tiempo de proceso en base de datos
            self.common_functions.save_process_time(st, event = str(model_name)+"_evaluate_model", description = "Time for model to be evaluated in test dataset")
            if(process_evaluate_result["status"] == True):
                del(process_evaluate_result)
                return {"status":True, "result":""}
            else:
                del(process_evaluate_result)
                return {"status":False, "result":"no se pudo ejecutar correctamente content_explicit"}
        else:
            print("decide what to do")
            #result_model.save(self.models_path+model)

        return {"status":True, "result":""}

    # Popularity Model
    def custom_popularity(df):
        try:
            # Se agrupan todos los eventos que ha tenido un usuario con un item y se suman los valores de target para esa relación
            df = df \
                            .groupby(['user_id', 'item_id'])['target'].sum() \
                            .apply(common.smooth_user_preference).reset_index()

            # Se suman los valores de target en todos los usuarios para cada item 
            df = df.groupby('item_id')['target'].sum().sort_values(ascending=False).reset_index()
            df['Rank'] = df['target'].rank(ascending=0, method='first')

            return {"status":True, "result":df}
        except ValueError as e:
            return {"status":False, "result":"There was an error trying to compute custom_popularity. Error: "+str(e)}