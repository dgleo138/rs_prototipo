#!~/dider/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
import os
import errno
import math
from sqlalchemy.sql import select
from sqlalchemy import Table
from sqlalchemy import MetaData
import pandas as pd
from collections import defaultdict

import sqlalchemy as sql
from sqlalchemy import create_engine
from timeit import default_timer
start = default_timer()
import datetime
import configparser

class common:

    def __init__(self, client_name):
        # Nombre del cliente para identificar sobre cuales archivos se ejecutaran los procesos
        self.client_name = client_name+'/'

        # Archivo de configuracion
        config = configparser.ConfigParser()
        config.sections()
        if os.path.isfile(str('../Datasets/'+self.client_name)+'config.ini'):
            with open(str('../Datasets/'+self.client_name)+'config.ini') as config_parser_fp:
                config.read_file(config_parser_fp)

        self.database_path = "../Datasets/"+str(self.client_name)+"database/input_data/"
        self.sql_db = sql.create_engine('sqlite:///'+self.database_path+"db.sql", encoding='utf-8')
        self.sql_db.raw_connection().connection.text_factory = str

    # ------------------------------------
    def initialize(self, data_path,data_path_backup,database_path,valid_data_directories):
        # Validate if defined data_path directory exist
        if(os.path.isdir(data_path) == False):
            return {"status": False, "result":"Defined data_path directory doesn't exist"}
        # Validate if defined data_path_backup directory exist
        if(os.path.isdir(data_path_backup) == False):
            return {"status": False, "result":"Defined data_path_backup directory doesn't exist"}
        # Validate if defined database_path directory exist
        if(os.path.isdir(database_path) == False):
            return {"status": False, "result":"Defined database_path directory doesn't exist"}

        # Validate if required directories exist in defined data_path. if not, create directories
        for data_dir in valid_data_directories:
            if(os.path.isdir(data_path+str(data_dir)) == False):
                try:
                    os.makedirs(data_path+str(data_dir))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        return {"status": False, "result":e}
                    
            if(os.path.isdir(data_path_backup+str(data_dir)) == False):
                try:
                    os.makedirs(data_path_backup+str(data_dir))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        return {"status": False, "result":e}

            if(os.path.isdir(database_path+"output") == False):
                try:
                    os.makedirs(database_path+"output")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        return {"status": False, "result":e}

            if(os.path.isdir(database_path+"output/data_split") == False):
                try:
                    os.makedirs(database_path+"output/data_split")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        return {"status": False, "result":e}

            if(os.path.isdir(database_path+"output/model_recs") == False):
                try:
                    os.makedirs(database_path+"output/model_recs")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        return {"status": False, "result":e}

        return {"status": True, "result":"Done"}

    # ------------------------------------
    def smooth_user_preference(self, x):
        return math.log(1+x, 2)

    # ------------------------------------
    def validate_available_sql_data(self, table_name,sql_db):
        if not sql_db.dialect.has_table(sql_db, table_name):
            return False
        else:
            s = select(["count(*) FROM "+str(table_name)])
            res = sql_db.execute(s)
            rows = res.fetchall()
            if rows==0:
                return False
            else:
                return True

    # ------------------------------------
    def drop_table_sql(self, table_name,sql_db):
        if(self.validate_available_sql_data(table_name,sql_db)):
            meta = MetaData()
            table = Table(table_name,meta)
            table.drop(sql_db)

    # ------------------------------------
    def is_set_property(self, config, property_set):
        key_map = config["key_map"]
        if((property_set in key_map) and (key_map[property_set] != "")):
            return True
        else:
            return False

    # ------------------------------------
    def get_items_interacted(self, person_id, interactions_df):
        # Get the user's data and merge in the item information.
        interacted_items = interactions_df.loc[person_id]['item_id']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])



    def save_process_time(self, st, event = "", description = ""):
        runtime = default_timer() - st
        print (str(description)+". Time:", round(runtime,2))
        # Almacenar tiempo de proceso en base de datos
        new_entry= pd.DataFrame(columns=['event','description','process_time','datetime'])
        new_entry = new_entry.append({'event':event,'description': description,"process_time":round(runtime,2), "datetime":datetime.datetime.now()}, ignore_index=True)
        new_entry.to_sql("time_results", self.sql_db, if_exists='append')
        del(new_entry)

    def reset_result_tables(self, sql_db):
        # Eliminar tablas temporales en base de datos
        self.drop_table_sql("popularity_results",sql_db)
        self.drop_table_sql("content_results",sql_db)
        self.drop_table_sql("collaborative_results",sql_db)
        self.drop_table_sql("matrix_results",sql_db)

    def get_available_data(self, sql_db):
        available_data = []
        # DATOS DISPONIBLES
        # Validar en base de datos si existen datos para explicit, implicit, explicit_review, user_content, item_content
        # En el caso de que exista la información append to available_data
        for name in ("explicit", "implicit", "explicit_review", "user_content", "item_content"): 
            data_table_result = self.validate_available_sql_data(name,sql_db)
            if(data_table_result == True):
                available_data.append(name)
        return available_data

    def get_user_content(self, sql_db):
        available_data = self.get_available_data(sql_db)
        # OBTENER INFORMACIÓN DE LOS USUARIOS
        # Validar si existe información adicional de los usuarios: user_content
        if("user_content" in available_data):
            # cargar la información en un dataframe
            user_content = pd.read_sql_query('select * from user_content;', sql_db, index_col='index')
        else:
            user_content = None
        return user_content

    def get_item_content(self, sql_db, database_path):
        available_data = self.get_available_data(sql_db)
        item_content_weights = 'auto'
        # Validar si existe información adicional de los items: item_content
        if("item_content" in available_data): 
            # cargar la información en un dataframe
            item_content = pd.read_sql_query('select * from item_content;', sql_db, index_col='index')

            # EXTRAER CONTENT WEIGHTS SI EXISTEN
            if(os.path.exists(database_path+"item_content_weights.json")):
                try:
                    # Obtener archivo con weights para el contenido del item
                    with open(database_path+"item_content_weights.json") as json_file:  
                        item_content_weights = json.load(json_file)
                except ValueError as e:
                    #return {"status":False, "result":"There was an error trying to load item_content_weights.json. error:"+str(e)}
                    print("There was an error trying to load item_content_weights.json. error:"+str(e))
            else:
                item_content_weights = 'auto'
        else:
            item_content = None

        return item_content, item_content_weights

    def get_top_n(self, predictions, n=10):
        '''Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n


    def precision_recall_at_k(self, predictions, k=10, threshold=3.5):
        '''Return precision and recall at k metrics for each user.'''

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls



    def precision_recall_at_k_hybrid(self, user_est_true, k=10, threshold=3.5):
        '''Return precision and recall at k metrics for each user.'''
        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls