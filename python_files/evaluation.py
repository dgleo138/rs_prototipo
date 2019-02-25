#!~/dider/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import time
from sklearn.model_selection import train_test_split
import configparser
import os
from surprise import accuracy

# Import local scripts
from common import common


class evaluation:

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

        self.database_path = "../Datasets/"+str(self.client_name)+"database/input_data/"
        self.explicit_rating_threshold_for_test = float(config['SAMPLING']['explicit_rating_threshold_for_test'])
        self.common_functions = common(self.client_name)


    def surprise_process_evaluate(self, predictions, knowledge_source, model_name, results_name, execution_time, k_recommend, sql_db, k_fold, verbose_switch = False, is_surprise=False):
        
        try:
            precisions, recalls = self.common_functions.precision_recall_at_k(predictions, k=k_recommend, threshold=self.explicit_rating_threshold_for_test)
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)
            precision_mean = sum(prec for prec in precisions.values()) / len(precisions)
            recall_mean = sum(rec for rec in recalls.values()) / len(recalls)
            model_results= pd.DataFrame(columns=['knowledge_source','model','kfold','precision','recall','rmse','mae','time'])            
            model_results = model_results.append({'knowledge_source':knowledge_source,'model': model_name,"kfold":k_fold,"precision":float(precision_mean), "recall":float(recall_mean), 'rmse':rmse, 'mae':mae, "time":execution_time}, ignore_index=True)
            print(model_results)
            model_results.to_sql(results_name, sql_db, if_exists='append')
            del(model_results)
            del(precision_mean)
            del(recall_mean)
            del(results_name)
            del(model_name)
            return {"status":True, "result":""}
        except ValueError as e:
            return {"status":False, "result":"There was an error trying to process and evaluate algorithm results for "+str(model_name)+". Error: "+str(e)}
        
    def surprise_hybrid_evaluate(self, predictions_list, predictions, knowledge_source, model_name, results_name, execution_time, k_recommend, sql_db, k_fold, verbose_switch = False, is_surprise=False):
        
        try:
            precisions, recalls = self.common_functions.precision_recall_at_k_hybrid(predictions, k=k_recommend, threshold=self.explicit_rating_threshold_for_test)
            rmse = accuracy.rmse(predictions_list)
            mae = accuracy.mae(predictions_list)
            precision_mean = sum(prec for prec in precisions.values()) / len(precisions)
            recall_mean = sum(rec for rec in recalls.values()) / len(recalls)
            print(rmse)
            print(mae)
            model_results= pd.DataFrame(columns=['knowledge_source','model','kfold','precision','recall','rmse','mae','time'])
            model_results = model_results.append({'knowledge_source':knowledge_source,'model': model_name,"kfold":k_fold,"precision":float(precision_mean), "recall":float(recall_mean), 'rmse':rmse, 'mae':mae, "time":execution_time}, ignore_index=True)
            print(model_results)
            model_results.to_sql(results_name, sql_db, if_exists='append')
            del(model_results)
            del(precision_mean)
            del(recall_mean)
            del(results_name)
            del(model_name)
            return {"status":True, "result":""}
        except ValueError as e:
            return {"status":False, "result":"There was an error trying to process and evaluate algorithm results for "+str(model_name)+". Error: "+str(e)}
        


