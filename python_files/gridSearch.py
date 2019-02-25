from timeit import default_timer
start = default_timer()
import configparser
from surprise.model_selection import GridSearchCV
import sqlalchemy as sql
from sqlalchemy import create_engine
import os
import pandas as pd
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

from common import common


class gridSearch:

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
        self.sql_db = sql.create_engine('sqlite:///'+self.database_path+"db.sql", encoding='utf-8')
        self.sql_db.raw_connection().connection.text_factory = str
        self.common_functions = common(self.client_name)



    def parameters_selection(self, data, model_type, knowledge, new=False):

        # Seleccion de modelo a utilizar
        if("surprise_svd" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('svd_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('svd_params',self.sql_db) == False):
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

                selected_params= pd.DataFrame(columns=['knowledge','algorithm','svd_n_factors','svd_n_epochs','svd_init_std_dev','svd_lr_all','svd_reg_all'])

                #Grid search SVD
                if(svd_grid_search == True):
                    print("STARTING SVD grid search")
                    param_grid = {'n_factors': [80, 100, 120], 'n_epochs': [10, 20, 30], 'init_std_dev': [0.1, 0.3, 0.5], 'lr_all': [0.005, 0.05, 0.1], 'reg_all': [0.02, 0.002, 0.1]}
                    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[svd_grid_metric]
                    svd_n_factors = int(selected["n_factors"])
                    svd_n_epochs = int(selected["n_epochs"])
                    svd_init_std_dev = float(selected["init_std_dev"])
                    svd_lr_all = float(selected["lr_all"])
                    svd_reg_all = float(selected["reg_all"])
                    print("DONE SVD grid search")

                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'svd','svd_n_factors':svd_n_factors,'svd_n_epochs':svd_n_epochs,'svd_init_std_dev':svd_init_std_dev,'svd_lr_all':svd_lr_all,'svd_reg_all':svd_reg_all}, ignore_index=True)
                selected_params.to_sql('svd_params', self.sql_db, if_exists='append')

        if("surprise_SVDpp" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('svdpp_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('svdpp_params',self.sql_db) == False):
                # Obtener valores de configuracion
                svdpp_grid_search = self.config['SURPRISE_SVDPP'].getboolean('svdpp_grid_search')
                svdpp_grid_metric = self.config['SURPRISE_SVDPP']['svdpp_grid_metric']
                svdpp_n_factors = int(self.config['SURPRISE_SVDPP']['svdpp_n_factors'])
                svdpp_n_epochs = int(self.config['SURPRISE_SVDPP']['svdpp_n_epochs'])
                svdpp_init_mean = float(self.config['SURPRISE_SVDPP']['svdpp_init_mean'])
                svdpp_init_std_dev = float(self.config['SURPRISE_SVDPP']['svdpp_init_std_dev'])
                svdpp_lr_all = float(self.config['SURPRISE_SVDPP']['svdpp_lr_all'])
                svdpp_reg_all = float(self.config['SURPRISE_SVDPP']['svdpp_reg_all'])

                selected_params= pd.DataFrame(columns=['knowledge','algorithm','svdpp_n_factors','svdpp_n_epochs','svdpp_init_std_dev','svdpp_lr_all','svdpp_reg_all'])

                #Grid search SVDpp
                if(svdpp_grid_search == True):
                    param_grid = {'n_factors': [20, 30, 40], 'n_epochs': [10, 20, 30], 'init_std_dev': [0.1, 0.3, 0.5], 'lr_all': [0.007, 0.07, 0.1], 'reg_all': [0.02, 0.002, 0.1]}
                    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[svdpp_grid_metric]
                    svdpp_n_factors = int(selected["n_factors"])
                    svdpp_n_epochs = int(selected["n_epochs"])
                    svdpp_init_std_dev = float(selected["init_std_dev"])
                    svdpp_lr_all = float(selected["lr_all"])
                    svdpp_reg_all = float(selected["reg_all"])
                
                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'svdpp','svdpp_n_factors':svdpp_n_factors,'svdpp_n_epochs':svdpp_n_epochs,'svdpp_init_std_dev':svdpp_init_std_dev,'svdpp_lr_all':svdpp_lr_all,'svdpp_reg_all':svdpp_reg_all}, ignore_index=True)
                selected_params.to_sql('svdpp_params', self.sql_db, if_exists='append')

        if("surprise_NMF" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('nmf_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('nmf_params',self.sql_db) == False):
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

                selected_params= pd.DataFrame(columns=['knowledge','algorithm','nmf_n_factors','nmf_n_epochs','nmf_reg_pu','nmf_reg_qi','nmf_init_low'])

                #Grid search NMF
                if(nmf_grid_search == True):
                    param_grid = {'n_factors': [10, 15, 20], 'n_epochs': [40, 50, 60], 'reg_pu': [0.006, 0.06, 0.1], 'reg_qi': [0.006, 0.06, 0.1], 'init_low': [0, 0.1, 0.2]}
                    gs = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[nmf_grid_metric]
                    nmf_n_factors = int(selected["n_factors"])
                    nmf_n_epochs = int(selected["n_epochs"])
                    nmf_reg_pu = float(selected["reg_pu"])
                    nmf_reg_qi = float(selected["reg_qi"])
                    nmf_init_low = float(selected["init_low"])
                
                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'nmf','nmf_n_factors':nmf_n_factors,'nmf_n_epochs':nmf_n_epochs,'nmf_reg_pu':nmf_reg_pu,'nmf_reg_qi':nmf_reg_qi,'nmf_init_low':nmf_init_low}, ignore_index=True)
                selected_params.to_sql('nmf_params', self.sql_db, if_exists='append')

        if("surprise_KNNBasic" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('knnbasic_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('knnbasic_params',self.sql_db) == False):
                # Obtener valores de configuracion
                knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
                knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
                knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
                knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

                selected_params= pd.DataFrame(columns=['knowledge','algorithm','knn_k','knn_min_k'])

                #Grid search KNN
                if(knn_grid_search == True):
                    param_grid = {'k': [20, 40, 60], 'min_k': [1, 2, 3, 4]}
                    gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[knn_grid_metric]
                    knn_k = int(selected["k"])
                    knn_min_k = int(selected["min_k"])
                
                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'knnbasic','knn_k':knn_k,'knn_min_k':knn_min_k}, ignore_index=True)
                selected_params.to_sql('knnbasic_params', self.sql_db, if_exists='append')
                
        if("surprise_KNNWithMeans" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('knnwithmeans_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('knnwithmeans_params',self.sql_db) == False):
                # Obtener valores de configuracion
                knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
                knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
                knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
                knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

                selected_params= pd.DataFrame(columns=['knowledge','algorithm','knn_k','knn_min_k'])

                #Grid search KNN
                if(knn_grid_search == True):
                    param_grid = {'k': [20, 40, 60], 'min_k': [1, 2, 3, 4]}
                    gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[knn_grid_metric]
                    knn_k = int(selected["k"])
                    knn_min_k = int(selected["min_k"])

                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'knnwithmeans','knn_k':knn_k,'knn_min_k':knn_min_k}, ignore_index=True)
                selected_params.to_sql('knnwithmeans_params', self.sql_db, if_exists='append')

        if("surprise_KNNWithZScore" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('knnwithzscore_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('knnwithzscore_params',self.sql_db) == False):
                # Obtener valores de configuracion
                knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
                knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
                knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
                knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

                selected_params= pd.DataFrame(columns=['knowledge','algorithm','knn_k','knn_min_k'])

                #Grid search KNN
                if(knn_grid_search == True):
                    param_grid = {'k': [20, 40, 60], 'min_k': [1, 2, 3, 4]}
                    gs = GridSearchCV(KNNWithZScore, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[knn_grid_metric]
                    knn_k = int(selected["k"])
                    knn_min_k = int(selected["min_k"])
                
                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'knnwithzscore','knn_k':knn_k,'knn_min_k':knn_min_k}, ignore_index=True)
                selected_params.to_sql('knnwithzscore_params', self.sql_db, if_exists='append')

        if("surprise_KNNBaseline" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('knnbaseline_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('knnbaseline_params',self.sql_db) == False):
                # Obtener valores de configuracion
                knn_k = int(self.config['SURPRISE_KNN']['knn_k'])
                knn_min_k = int(self.config['SURPRISE_KNN']['knn_min_k'])
                knn_grid_search = self.config['SURPRISE_KNN'].getboolean('knn_grid_search')
                knn_grid_metric = self.config['SURPRISE_KNN']['knn_grid_metric']

                selected_params= pd.DataFrame(columns=['knowledge','algorithm','knn_k','knn_min_k'])

                #Grid search KNN
                if(knn_grid_search == True):
                    param_grid = {'k': [20, 40, 60], 'min_k': [1, 2, 3, 4]}
                    gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[knn_grid_metric]
                    knn_k = int(selected["k"])
                    knn_min_k = int(selected["min_k"])
                
                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'knnbaseline','knn_k':knn_k,'knn_min_k':knn_min_k}, ignore_index=True)
                selected_params.to_sql('knnbaseline_params', self.sql_db, if_exists='append')
                
        if("surprise_CoClustering" in model_type):
            if(new == True):
                self.common_functions.drop_table_sql('coclustering_params',self.sql_db)
            if(self.common_functions.validate_available_sql_data('coclustering_params',self.sql_db) == False):
                # Obtener valores de configuracion
                cc_grid_search = self.config['SURPRISE_COCLUSTERING'].getboolean('cc_grid_search')
                cc_grid_metric = self.config['SURPRISE_COCLUSTERING']['cc_grid_metric']
                cc_n_cltr_u = int(self.config['SURPRISE_COCLUSTERING']['cc_n_cltr_u'])
                cc_n_cltr_i = int(self.config['SURPRISE_COCLUSTERING']['cc_n_cltr_i'])
                cc_n_epochs = int(self.config['SURPRISE_COCLUSTERING']['cc_n_epochs'])
                
                selected_params= pd.DataFrame(columns=['knowledge','algorithm','cc_n_cltr_u','cc_n_cltr_i','cc_n_epochs'])

                #Grid search CoClustering
                if(cc_grid_search == True):
                    param_grid = {'n_cltr_u': [3, 5, 7], 'n_cltr_i': [3, 5, 7], 'n_epochs': [15, 20, 25]}
                    gs = GridSearchCV(CoClustering, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=10)
                    gs.fit(data)
                    selected = gs.best_params[cc_grid_metric]
                    cc_n_cltr_u = int(selected["n_cltr_u"])
                    cc_n_cltr_i = int(selected["n_cltr_i"])
                    cc_n_epochs = int(selected["n_epochs"])
                
                selected_params=selected_params.append({'knowledge':knowledge,'algorithm':'coclustering','cc_n_cltr_u':cc_n_cltr_u,'cc_n_cltr_i':cc_n_cltr_i,'cc_n_epochs':cc_n_epochs}, ignore_index=True)
                selected_params.to_sql('coclustering_params', self.sql_db, if_exists='append')
