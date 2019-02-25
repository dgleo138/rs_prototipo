#!~/dider/bin/python
# -*- coding: UTF-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pandas as pd
import sqlalchemy as sql
from sqlalchemy import create_engine
from sqlalchemy.sql import select
from sqlalchemy import Table
from sqlalchemy import MetaData
from sklearn import preprocessing
import random
import numpy
import time
import os
from collections import defaultdict
import collections

# Import local scripts
from common import common
from evaluation import evaluation
import configparser
pd.set_option('display.max_columns', 500)

class hybrid:

    def __init__(self, client_name):
        # Nombre del cliente para identificar sobre cuales archivos se ejecutaran los procesos
        self.client_name = client_name
        self.config = configparser.ConfigParser()
        self.config.sections()
        if os.path.isfile(str('../Datasets/'+self.client_name)+'config.ini'):
            with open(str('../Datasets/'+self.client_name)+'config.ini') as config_parser_fp:
                self.config.read_file(config_parser_fp)

        self.database_path = "../Datasets/"+str(self.client_name)+"database/input_data/"
        self.sql_db = sql.create_engine('sqlite:///'+self.database_path+"db.sql")
        self.models_path = "../Datasets/"+str(self.client_name)+"database/models/"
        self.precision_weight = float(self.config['EVALUATION']['precision_weight'])
        self.recall_weight = float(self.config['EVALUATION']['recall_weight'])
        self.time_weight = float(self.config['EVALUATION']['time_weight'])
        self.rmse_weight = float(self.config['EVALUATION']['rmse_weight'])
        self.mae_weight = float(self.config['EVALUATION']['mae_weight'])
        self.common_functions = common(self.client_name)
        self.evaluation = evaluation(self.client_name)
        


    def select_best_results(self, knowledge_name="explicit", hybrid_list=False):
        array_result={}

        sql_db = self.sql_db
        # Se crea dataframe para almacenar los resultados finales por algoritmos
        final_results= pd.DataFrame(columns=['algorithm','model','precision','recall','rmse','mae','time','total'])
        # Se valida si existe cada tipo de resultado (pensado para diferentes fuentes de información)
        if(self.common_functions.validate_available_sql_data("collaborative_results",sql_db)):
            collaborative_results = pd.read_sql_query('select * from collaborative_results;', sql_db, index_col='index')
            array_result["collaborative"] = collaborative_results
        if(self.common_functions.validate_available_sql_data("popularity_results",sql_db)):
            popularity_results = pd.read_sql_query('select * from popularity_results;', sql_db, index_col='index')
            array_result["popularity"] = popularity_results
        if(self.common_functions.validate_available_sql_data("content_results",sql_db)):
            content_results = pd.read_sql_query('select * from content_results;', sql_db, index_col='index')
            array_result["content"] = content_results
        if(self.common_functions.validate_available_sql_data("mixed",sql_db)):
            mixed_results = pd.read_sql_query('select * from mixed;', sql_db, index_col='index')
            array_result["mixed"] = mixed_results

        for key in array_result:
            results=array_result[key]
            results=results[results["knowledge_source"]==knowledge_name]

            if(results.empty == False):
                indexed_dataset = results.set_index('model')
                # Obtener modelos unicos en el dataset
                res = enumerate(list(indexed_dataset.index.unique().values))
                partial_results= pd.DataFrame(columns=['algorithm','model','precision','recall','rmse','mae','time'])
                # Para cada usuario
                for r in res:
                    model_pd = results[results['model'] == r[1]]
                    precision_mean = model_pd.loc[:,"precision"].mean()
                    recall_mean = model_pd.loc[:,"recall"].mean()
                    time_mean = model_pd.loc[:,"time"].mean()
                    rmse_mean = model_pd.loc[:,"rmse"].mean()
                    mae_mean = model_pd.loc[:,"mae"].mean()
                    partial_results = partial_results.append({"algorithm":key,"model": r[1],"precision":precision_mean,"recall":recall_mean,'rmse':rmse_mean,'mae':mae_mean,"time":time_mean}, ignore_index=True)
                # Normalizar mae
                max_result_mae = partial_results["mae"].max()
                min_result_mae = partial_results["mae"].min()
                mirror_mae = 1
                if(max_result_mae == min_result_mae):
                    mirror_mae = max_result_mae
                    max_result_mae = 1
                    min_result_mae = 0
                partial_results["norm_mae"] = mirror_mae - ((partial_results["mae"] - min_result_mae)/(max_result_mae - min_result_mae))
                # Normalizar rmse
                max_result_rmse = partial_results["rmse"].max()
                min_result_rmse = partial_results["rmse"].min()
                mirror_rmse = 1
                if(max_result_rmse == min_result_rmse):
                    mirror_rmse = max_result_rmse
                    max_result_rmse = 1
                    min_result_rmse = 0
                partial_results["norm_rmse"] = mirror_rmse - ((partial_results["rmse"] - min_result_rmse)/(max_result_rmse - min_result_rmse))
                # Normalizar time
                max_result_time = partial_results["time"].max()
                min_result_time = partial_results["time"].min()
                mirror_time = 1
                if(max_result_time == min_result_time):
                    mirror_time = max_result_time
                    max_result_time = 1
                    min_result_time = 0
                partial_results["norm_time"] = mirror_time - ((partial_results["time"] - min_result_time)/(max_result_time - min_result_time))
                # Hallar total en todas las metricas, usando los pesos asignados por el proveedor en configuración
                partial_results["total"] = (partial_results["precision"]*self.precision_weight)+(partial_results["recall"]*self.recall_weight)+(partial_results["norm_rmse"]*self.rmse_weight)+(partial_results["norm_mae"]*self.mae_weight)+(partial_results["norm_time"]*self.time_weight)
                final_results = final_results.append(partial_results, ignore_index=True)
        results = final_results.sort_values("total",ascending=False)
        print(results)

        # Si ya existe un resultado total almacenado en base de datos para la fuente de informacion iterada
        if(self.common_functions.validate_available_sql_data(knowledge_name+"_totals",sql_db) == True):
            # Se elimina el registro existente para el proveedor
            self.common_functions.drop_table_sql(knowledge_name+"_totals",sql_db)
        # Se almacena el nuevo registro (total)
        results.to_sql(knowledge_name+"_totals", sql_db, if_exists='append')

        algorithms_hyb_list = list()
        # Si el parametro hybrid_list esta activo, se devuelve un listado con los algoritmos que seran hibridados segun el valor total y el umbral de hibridacion definido en configuracion
        if(hybrid_list == True):
            # Seleccionar algoritmos para los que su puntuacion este sobre el threshold definido
            hybridation_threshold = float(self.config['HYBRIDATION']['hybridation_threshold'])
            max_for_hybrid = int(self.config['HYBRIDATION']['max_for_hybrid'])
            hybridation_results = results[results["total"] >= hybridation_threshold]
            if(len(hybridation_results.index) < 2):
                hybridation_results = results.head(2)
            elif(len(hybridation_results.index) > max_for_hybrid):
                hybridation_results = results.head(max_for_hybrid)
            for index, item in hybridation_results.iterrows():
                algorithms_hyb_list.append(item["model"])
            print(algorithms_hyb_list)
            return algorithms_hyb_list
        else:
            # En caso contrario se devuelve una lista vacia
            return algorithms_hyb_list


    def hybrid_algorithms(self, algo_1, algo_2, kfold, knowledge, k_recommend):

        # Obtener los totales de cada algoritmo para usar como peso en la hibridacion
        alg_actual_result = pd.read_sql_query('select * from '+knowledge+'_totals;', self.sql_db, index_col='index')
        res_alg_1 = alg_actual_result[alg_actual_result["model"] == algo_1].iloc[0]['total']
        res_alg_2 = alg_actual_result[alg_actual_result["model"] == algo_2].iloc[0]['total']
        total_both_algorithms = res_alg_1 + res_alg_2
        weight_algo_1 = res_alg_1/total_both_algorithms
        weight_algo_2 = res_alg_2/total_both_algorithms

        training_time = (alg_actual_result[alg_actual_result["model"] == algo_1].iloc[0]['time'] + alg_actual_result[alg_actual_result["model"] == algo_2].iloc[0]['time'])

        model_name = algo_1+"_"+algo_2

        hybridation = pd.DataFrame(columns=['name','model1','model2'])
        hybridation = hybridation.append({"name":model_name,"model1": algo_1,"model2":algo_2}, ignore_index=True)
        hybridation.to_sql(knowledge+"_hybrids", self.sql_db, if_exists='replace')

        for k in range(1,(kfold+1)): 
            # Si existe el archivo
            if(os.path.exists(self.models_path+algo_1+"/predictions/"+str(k)) == True):
                # cargar predicciones del argoritmo 1
                alg_1_predictions = pd.read_csv(self.models_path+algo_1+"/predictions/"+str(k)+"/predictions.csv", encoding='latin-1', sep=str(u';').encode('utf-8'))
            else:
                print("missed 1: "+self.models_path+algo_1+"/predictions/"+str(k)+"/predictions.csv")
                return {"status": False, "result":"There was not possible to find predictions file for "+self.models_path+algo_1+"/predictions/"+str(kfold)}

            if(os.path.exists(self.models_path+algo_2+"/predictions/"+str(k)) == True):
                # cargar predicciones del argoritmo 1
                alg_2_predictions = pd.read_csv(self.models_path+algo_2+"/predictions/"+str(k)+"/predictions.csv", encoding='latin-1', sep=str(u';').encode('utf-8'))
            else:
                print("missed 2: "+self.models_path+algo_2+"/predictions/"+str(k)+"/predictions.csv")
                return {"status": False, "result":"There was not possible to find predictions file for "+self.models_path+algo_2+"/predictions/"+str(kfold)}

            user_est_true = defaultdict(list)
            Point = collections.namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'], verbose=True)
            predictions_list = []
            # Hibridar los algoritmos
            for index, item in alg_1_predictions.iterrows():
                alg_2_equal = alg_2_predictions[(alg_2_predictions["user_id"]==item["user_id"]) & (alg_2_predictions["item_id"]==item["item_id"])]
                new_est = (alg_2_equal.iloc[0]['est'] * weight_algo_2) + (item['est'] * weight_algo_1)
                user_est_true[item['user_id']].append((new_est, item['r_ui']))
                predictions_list.append(Point(uid=unicode(str(item['user_id'])), iid=unicode(str(item['item_id'])), r_ui=float(item['r_ui']), est=float(new_est), details={u'was_impossible': False}))
            print("fold: "+str(k))
            # Evaluar hibridacion
            self.evaluation.surprise_hybrid_evaluate(predictions_list, user_est_true, knowledge, model_name, "collaborative_results", training_time, k_recommend, self.sql_db, k)
        del(alg_1_predictions)
        del(alg_2_predictions)

    def select_best_algorithm_knowledge(self, knowledge_name):

        knowledge_result = pd.read_sql_query('select * from '+knowledge_name+'_totals;', self.sql_db, index_col='index')
        knowledge_result = knowledge_result.sort_values("total",ascending=False)

        # Obtener mejor modelo en la primera posicion
        best_model = knowledge_result["model"].iloc[0]
        # Preguntar si el modelo seleccionado existe en la tabla de hibridaciones
        if(self.common_functions.validate_available_sql_data(knowledge_name+"_hybrids",self.sql_db) == True):
            hybrids_result = pd.read_sql_query('select * from '+knowledge_name+'_hybrids;', self.sql_db, index_col='index')
            hybrid_model = hybrids_result[hybrids_result["name"] == best_model]
            if(hybrid_model.empty == False):
                model_1 = hybrid_model["model1"]
                model_2 = hybrid_model["model2"]
            else:
                model_1 = best_model
                model_2 = ""
        else:
            model_1 = best_model
            model_2 = ""

        best_results= pd.DataFrame(columns=['algorithm','model_1','model_2'])
        best_results = best_results.append({'algorithm':knowledge_result["algorithm"].iloc[0],'model_1': model_1,'model_2':model_2}, ignore_index=True)
        # Eliminar tabla si existe
        if(self.common_functions.validate_available_sql_data(knowledge_name+"_selected_algorithms",self.sql_db) == True):
            self.common_functions.drop_table_sql(knowledge_name+"_selected_algorithms",self.sql_db)
        best_results.to_sql(knowledge_name+"_selected_algorithms", self.sql_db, if_exists='append')
        print(best_results)
        del(best_results)

    def get_significance(self, model_1, model_2, metric):

        # Se crea dataframe para almacenar los resultados finales por algoritmos
        final_results= pd.DataFrame(columns=['model','k','precision','recall','rmse','mae','time','norm_mae','norm_rmse','norm_time','total'])

        # Se obtienen todos los datos disponibles
        collaborative_results = pd.read_sql_query('select * from collaborative_results;', self.sql_db, index_col='index')

        # Validar que el modelo 1 exista en los datos disponibles
        model_1_validation = collaborative_results[collaborative_results["model"] == model_1]
        if(model_1_validation.empty == True):
            return {"status":False, "result":"Model 1: "+str(model_1)+" doesn't exist in available data"}
        
        # Validar que el modelo 2 exista en los datos disponibles
        model_2_validation = collaborative_results[collaborative_results["model"] == model_2]
        if(model_2_validation.empty == True):
            return {"status":False, "result":"Model 1: "+str(model_2)+" doesn't exist in available data"}
        

        collaborative_results_paired = collaborative_results[(collaborative_results["model"] == model_1) | (collaborative_results["model"] == model_2)]
        
        # Obtener los valores maximo y minimo de todas las iteraciones de los modelos
        max_result_mae = collaborative_results["mae"].max()
        min_result_mae = collaborative_results["mae"].min()
        mirror_mae = 1
        if(max_result_mae == min_result_mae):
            mirror_mae = max_result_mae
            max_result_mae = 1
            min_result_mae = 0

        # Obtener los valores maximo y minimo de todas las iteraciones de los modelos
        max_result_rmse = collaborative_results["rmse"].max()
        min_result_rmse = collaborative_results["rmse"].min()
        mirror_rmse = 1
        if(max_result_rmse == min_result_rmse):
            mirror_rmse = max_result_rmse
            max_result_rmse = 1
            min_result_rmse = 0
        
        # Obtener los valores maximo y minimo de todas las iteraciones de los modelos
        max_result_time = collaborative_results["time"].max()
        min_result_time = collaborative_results["time"].min()
        mirror_time = 1
        if(max_result_time == min_result_time):
            mirror_time = max_result_time
            max_result_time = 1
            min_result_time = 0

        if(collaborative_results_paired.empty == False):
            indexed_dataset = collaborative_results_paired.set_index('model')
            # Obtener modelos unicos en el dataset
            res = enumerate(list(indexed_dataset.index.unique().values))
            
            # Para cada usuario
            for r in res:
                partial_results= pd.DataFrame(columns=['model','k','precision','recall','rmse','mae','time','norm_mae','norm_rmse','norm_time','total'])
                model_each_k = collaborative_results_paired[collaborative_results_paired["model"]==r[1]]

                for index, model_k in model_each_k.iterrows():
                    norm_mae = mirror_mae - ((model_k["mae"] - min_result_mae)/(max_result_mae - min_result_mae))
                    norm_rmse = mirror_rmse - ((model_k["rmse"] - min_result_rmse)/(max_result_rmse - min_result_rmse))
                    norm_time = mirror_time - ((model_k["time"] - min_result_time)/(max_result_time - min_result_time))
                    partial_results = partial_results.append({'model':r[1],'k':model_k["kfold"],'precision':model_k["precision"],'recall':model_k["recall"],'rmse':model_k["rmse"],'mae':model_k["mae"],'time':model_k["time"],'norm_mae':norm_mae,'norm_rmse':norm_rmse,'norm_time':norm_time}, ignore_index=True)
                    partial_results["total"] = (partial_results["precision"]*self.precision_weight)+(partial_results["recall"]*self.recall_weight)+(partial_results["norm_rmse"]*self.rmse_weight)+(partial_results["norm_mae"]*self.mae_weight)+(partial_results["norm_time"]*self.time_weight)
                partial_results = partial_results.sort_values("k",ascending=True)
                final_results = final_results.append(partial_results, ignore_index=True)
            
            print(final_results)

            # Obtener los valores de la metrica a usr en el test estadistico
            samples_model_1 = final_results[final_results["model"] == model_1]
            samples_model_1 = samples_model_1[metric].values
            print(samples_model_1)
            samples_model_2 = final_results[final_results["model"] == model_2]
            samples_model_2 = samples_model_2[metric].values
            print(samples_model_2)

            from scipy import stats
            significance_result = stats.ttest_ind(samples_model_1,samples_model_2)
            print(significance_result)
