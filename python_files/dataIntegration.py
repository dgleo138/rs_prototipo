#!~/dider/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import os
import json
import numpy as np
import shutil
import glob
from textblob import TextBlob
from timeit import default_timer
start = default_timer()
import datetime
import sqlalchemy as sql
from sqlalchemy import create_engine
import configparser

class dataStandardization:

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
        self.sql_db = sql.create_engine('sqlite:///'+self.database_path+"db.sql", encoding='utf-8')
        self.sql_db.raw_connection().connection.text_factory = str

    # ------------------------------------
    # Funcion para realizar el proceso de estandarizacion de los datos
    def data_etl(self,valid_data_conf_names, data_path, data_path_backup, sql_db, database_path, storage_type="fs", action="update"):
        # Si el tipo de almacenamiento es file system (fs)
        if(storage_type == "fs"):
            # Variable para almacenar los tipos de datos que se han recorrido con exito
            acepted_data_type_dic = []
            # Recorrer los archivos de configuración validos y buscar en data_path si existen
            for conf_name in valid_data_conf_names:
                # Si existe la ruta al archivo de configuración
                if(os.path.exists(data_path+str(conf_name))):
                    #cargar configuración, recorrer archivos en el directorio, convertir a dataframe y guardar registros
                    try:
                        # Obtener archivo de configuración de la fuente de datos iterada
                        with open(data_path+str(conf_name), 'r') as f:
                            config = json.load(f)
                    except ValueError as e:
                        return {"status":False, "result":"There was an error trying to load "+str(conf_name)+": try removing ',' character where no next key follows. error: "+str(e)}
                    
                    # Obtener la configuración para el tipo de archivo
                    file_config = config["file_type_information"]
                    
                    # Preguntar si hay archivos en el directorio correspondiente al tipo de archivo iterado
                    if len(os.listdir(data_path+str(config["input_type"])+"/")) != 0:
                        print("STARTING extraction from data source: "+config["input_type"])
                        # Tomar tiempo de inicio
                        st = default_timer()
                        
                        # Si el tipo de archivos tienen un formato diferente a json, por defecto tratados como csv
                        if(config["is_json"] == False):
                            try:
                                # Cuando headers = True
                                if(file_config["csv_headers"] == True):
                                    # Concatenar archivos en dataframe con headers
                                    df = pd.concat([pd.read_csv(f, encoding='latin1', sep=file_config["csv_separator"], error_bad_lines=False, index_col=False) for f in glob.glob(data_path+str(config["input_type"])+"/*")], ignore_index = True, axis=0)
                                # Cuando headers = False
                                else:
                                    # Concatenar archivos en dataframe sin headers
                                    df = pd.DataFrame(np.concatenate([pd.read_csv(f, encoding='latin1', sep=file_config["csv_separator"], error_bad_lines=False, index_col=False) for f in glob.glob(data_path+str(config["input_type"])+"/*")]))
                            except ValueError as e:
                                return {"status":False, "result":"There was an error trying to concat files to a dataframe from input type "+str(config["input_type"])+". error: "+str(e)}
                        
                        # Si el tipo de archivos tienen un formato json
                        else:
                            try:
                                # Concatenar archivos en dataframe
                                df = pd.concat([pd.read_json(f, encoding='latin1', lines=file_config["json_lines"]) for f in glob.glob(data_path+str(config["input_type"])+"/*")], ignore_index = True, axis=0)
                            except ValueError as e:
                                return {"status":False, "result":"There was an error trying to concat files to a dataframe from input type "+str(config["input_type"])+". error: "+str(e)}
                        
                        # Los archivos procesados se mueven a un directorio aparte para cuando se reciban nuevos archivos, no tener que procesar los anteriores nuevamnete. Por otro lado, esto permite tener un respaldo de los archivos procesados
                        # Mover los archivos del dataset ya procesados al directorio de backup.
                        source = data_path+str(config["input_type"])+"/"
                        dest1 = data_path_backup+str(config["input_type"])+"/"
                        files = os.listdir(source)
                        #for f in files:
                        #       shutil.move(source+f, dest1)
                        
                        # Validar, limpiar y organizar datos al formato requerido
                        result = self.defineKnowledge_sources(config, df)
                        
                        # Si el proceso de limpieza y validación fue exitoso
                        if(result["status"] == True):
                            
                            # Agregar los tipos de archivo completamente procesados para reporte final
                            acepted_data_type_dic.append(config["input_type"])
                            
                            # Obtener key_map desde el archivo de configuración
                            key_map = config['key_map']
                            
                            # Obtener el dataframe del resultado de limpieza y validación
                            df = result["data"]
                            
                            # Cambio de tipo de dato para columnas que se almacenan como tipo Object o tipo Mixed
                            types = df.apply(lambda x: pd.api.types.infer_dtype(x.values))
                            
                            if(config["is_json"] == True):
                                #print(types)
                                for col in types[types=='unicode'].index:
                                    df[col] = df[col].astype(str)
                                for col in types[types=='mixed'].index:
                                    df[col] = df[col].astype(str)

                            #SQLITE
                            # Guardar los datos del dataframe en persistencia para el cliente actual.
                            sql_db.text_factory = str
                            df.to_sql(str(config["input_type"]), sql_db, if_exists='append')
                            
                            print("ENDING extraction from data source: "+config["input_type"])
                            runtime = default_timer() - st
                            # Almacenar tiempo de proceso en base de datos
                            new_entry= pd.DataFrame(columns=['event','description','process_time','datetime'])
                            new_entry = new_entry.append({'event':"get_knowledge_"+config["input_type"],'description': "Tiempo en transformar y extraer la fuente de conocimiento: "+config["input_type"],"process_time":round(runtime,2), "datetime":datetime.datetime.now()}, ignore_index=True)
                            new_entry.to_sql("time_results", sql_db, if_exists='append')
                            del(new_entry)
                            del(result)
                            
                            # Casos especiales (para uso de otras fuentes de informacion)
                            # Para item_content preguntar si se declararon weights para cada caracteristica del item
                            if((config["input_type"] == "item_content") and ("content_weight" in key_map) and (key_map["content_weight"] != "")):
                                item_content_weights = key_map["content_weight"]
                                if (type(item_content_weights) is not dict):
                                    return {"status":False, "result":"content_weights value for item_content configuration must be a dictionary. the actual value is: "+str(type(item_content_weights))}
                                else:
                                    # Guardar los weights en persistencia
                                    item_c_json = json.dumps(item_content_weights)
                                    f = open(database_path+"item_content_weights.json","w")
                                    f.write(item_c_json)
                                    f.close()
                                    
                            # Para user_content preguntar si se declararon weights para cada caracteristica del user
                            if((config["input_type"] == "user_content") and("content_weight" in key_map) and (key_map["content_weight"] != "")):
                                user_content_weights = key_map["content_weight"]
                                if (type(user_content_weights) is not dict):
                                    return {"status":False, "result":"content_weights value for item_content configuration must be a dictionary. the actual value is: "+str(type(item_content_weights))}
                                else:
                                # Guardar los weights en persistencia
                                    user_c_json = json.dumps(user_content_weights)
                                    f = open(database_path+"user_content_weights.json","w")
                                    f.write(user_c_json)
                                    f.close()

                        else:
                            return {"status":False, "result":result["data"]}
                        return {"status":True, "result":"ETL process was completed for datasources: "+ str(acepted_data_type_dic)}
            
        else:
            return {"status":False, "result":"Defined type is not supported"}

    # ------------------------------------
    # Importar datasets (Batch)
    # Recorrer configurción y completar acción para cada tipo de archivo
    def defineKnowledge_sources(self, config, df):
        
        print(config["input_type"])
        
        # Se crean variables correspondientes a cada tipo de información
        implicit = None
        explicit = None
        explicit_review = None
        item_content = None
        user_content = None

        # Cargar archivo como dataframe

        # Interaction
        if(config['input_type']== "implicit"):
            status,implicit = self.get_interactions_dataframe(config, df)
            if(status == False):
                return {"status":False, "data":implicit}
            else:
                return {"status":True, "data":implicit}
            
        # explicit
        if(config['input_type']== "explicit"):
            status,explicit = self.get_explicit_dataframe(config, df)
            if(status == False):
                return {"status":False, "data":explicit}
            else:
                return {"status":True, "data":explicit}
        
        # explicit_review
        if(config['input_type']== "explicit_review"):
            status,explicit_review = self.get_explicit_review_dataframe(config, df)
            if(status == False):
                return {"status":False, "data":explicit_review}
            else:
                return {"status":True, "data":explicit_review}
            
        # item_content
        if(config['input_type']== "item_content"):
            status,item_content = self.get_item_content_dataframe(config, df)
            if(status == False):
                return {"status":False, "data":item_content}
            else:
                return {"status":True, "data":item_content}
            
        # user_content
        if(config['input_type']== "user_content"):
            status,user_content = self.get_user_content_dataframe(config, df)
            if(status == False):
                return {"status":False, "data":user_content}
            else:
                return {"status":True, "data":user_content}


    # ------------------------------------
    # Function to get interaction config and return dataframe
    def get_interactions_dataframe(self, config, df):
        interaction_required_fields = {
                    "user_id":False,
                    "item_id":False
                }
        
        fields = config['key_map']
        file_type_config = config['file_type_information']
        
        # Se obtienen las columnas declaradas en configuración
        field_names = []
        column_names_relation = {}
        for key in fields:
            if(fields[key] != ""):
                # Validar que los campos obligatorios esten definidos
                if(key in interaction_required_fields):
                    interaction_required_fields[key] = True
                if(key == "event_name"):
                    field_names.append("event_name")
                    column_names_relation["event_name"]=fields[key]
                if(key == "user_id"):
                    field_names.append("user_id")
                    column_names_relation["user_id"]=fields[key]
                if(key == "item_id"):
                    field_names.append("item_id")
                    column_names_relation["item_id"]=fields[key]
                if(key == "value"):
                    field_names.append("value")
                    column_names_relation["value"]=fields[key]
                if(key == "timestamp"):
                    field_names.append("timestamp")
                    column_names_relation["timestamp"]=fields[key]
                if(key == "event_side_features"):
                    side_features = fields["event_side_features"]
                    for feature in side_features:
                        field_names.append(feature)
                        column_names_relation[feature]=side_features[feature]
                        
        # Se valida que todos los campos obligatorios esten definidos
        for valid_iter in interaction_required_fields:
            if(interaction_required_fields[valid_iter] == False):
                return False, "Required field is not defined in configuration. Field: "+str(valid_iter), False
    
        # Cambiar/agregar nombres al dataframe
        for pair in column_names_relation:
            if(config["is_json"] == False):
                if(file_type_config["csv_headers"] == True):
                    df=df.rename(columns = {column_names_relation[pair]:pair})
                else:
                    df.columns.values[column_names_relation[pair]] = pair
            else:
                df=df.rename(columns = {column_names_relation[pair]:pair})
                
        # Extraer solo las columnas requeridas en el dataframe
        df = df[field_names]
        
        # Validar los tipos de datos para item_id, user_id
        # item_id

        d1 = df.apply(lambda x: pd.api.types.infer_dtype(x.values)) #df.dtypes.astype(str).to_dict()
        if(d1["item_id"] == "float64"):
            return False,"Values on item_id can only be of type String or Int. Float values found."
        # user_id
        if(d1["user_id"] == "float64"):
            return False,"Values on user_id can only be of type String or Int. Float values found."
        
        
        # Filtrar el contenido definido en configuración
        if(("remove_content" in fields) and (fields["remove_content"] != "")):
            remove_content = fields["remove_content"]
            for remove in remove_content:
                for remove_value in remove_content[remove]:
                    df = df[df[remove] != remove_value]

        # Validar que no existan campos de user_id o item_id vacios
        df = df.loc[pd.isnull(df.item_id) == False]
        df = df.loc[pd.isnull(df.user_id) == False]

        #valida los campos vacios en todas las columnas y agregar un valor dependiendo del tipo de columna
        for column in field_names:
            if(d1[column] == "object"):
                df[column].fillna('e', inplace=True)
            else:
                df[column].fillna(0, inplace=True)
        del(d1)

        # Si no existe la columna value, se debe agregar con el valor de 1 en todos los registros
        if(("value" not in field_names) and (fields["value"] == "")):
            df['value']=1
        else:
            # Agregar cero a los valores vacios en la columna values
            df['value'].fillna(0)
    
        # Obtener los pesos definidos en configuración para cada tipo de evento
        if(("event_weight" in fields) and (fields["event_weight"] != "") and ("event_name" in fields) and (fields["event_name"] != "")):
            weights = fields["event_weight"]
            df['eventStrength'] = df['event_name'].apply(lambda x: weights[x] if x in weights else 0)
        else:
            df['eventStrength']=1
        
        # Validar los tipos de datos para value y weight
        # value
        d1 = df.apply(lambda x: pd.api.types.infer_dtype(x.values)) #df.dtypes.astype(str).to_dict()
        if(("value" in field_names) and (d1["value"] == "object")):
            return False,"Values on column 'value' can only be of type Int or Float. String values found.", False
        # weight
        if(d1["eventStrength"] == "object"):
            return False,"Values on defined weights must be int or float. String values found.", False
        del(d1)
        
        # Multiplicar las columnas value y weight
        df['target'] = df.eventStrength * df.value
        
        # Se remueven las columnas 'value' y 'eventStrenght' del dataframe
        df = df.drop(['value', 'eventStrength'], axis=1)
        
        # Se remueve la columna 'event_name' del dataframe
        if('event_name' in column_names_relation):
            df = df.drop(['event_name'], axis=1)
        
        return True, df



    # ------------------------------------
    # Function to get exlicit config and return dataframe
    def get_explicit_dataframe(self, config, df):
        interaction_required_fields = {
                    "user_id":False,
                    "item_id":False,
                    "rating":False,
                }
        
        fields = config['key_map']
        file_type_config = config['file_type_information']

        # Obtene la configuración para la columna rating
        if(("rating_config" not in fields) or (fields["rating_config"] == "")):
            return False, "For explicit data is required to define rating_config property in configuration file"
        else:
            rating_config = fields["rating_config"]
            if(("min_rating" not in rating_config) or (rating_config["min_rating"] == "") or (type(rating_config["min_rating"]) is str)):
                return False, "rating_config parameter in explicit configuration file must contain a 'min_rating' key with an int value"
            if(("max_rating" not in rating_config) or (rating_config["max_rating"] == "") or (type(rating_config["max_rating"]) is str)):
                return False, "rating_config parameter in explicit configuration file must contain a 'max_rating' key with an int value"
            if(rating_config["min_rating"] >= rating_config["max_rating"]):
                return False, "'max_rating' parameter must be greater and different from 'min_rating' parameter" 
        
        # Se obtienen las columnas declaradas en configuración
        field_names = []
        column_names_relation = {}
        for key in fields:
            if(fields[key] != ""):
                # Validar que los campos obligatorios esten definidos
                if(key in interaction_required_fields):
                    interaction_required_fields[key] = True
                if(key == "user_id"):
                    field_names.append("user_id")
                    column_names_relation["user_id"]=fields[key]
                if(key == "item_id"):
                    field_names.append("item_id")
                    column_names_relation["item_id"]=fields[key]
                if(key == "rating"):
                    field_names.append("target")
                    column_names_relation["target"]=fields[key]
                if(key == "timestamp"):
                    field_names.append("timestamp")
                    column_names_relation["timestamp"]=fields[key]
                if(key == "event_side_features"):
                    side_features = fields["event_side_features"]
                    for feature in side_features:
                        field_names.append(feature)
                        column_names_relation[feature]=side_features[feature]
                        
        # Se valida que todos los campos obligatorios esten definidos
        for valid_iter in interaction_required_fields:
            if(interaction_required_fields[valid_iter] == False):
                return False, "Required field is not defined in configuration. Field: "+str(valid_iter)
        
        # Cambiar/agregar nombres al dataframe
        for pair in column_names_relation:
            if(config["is_json"] == False):
                if(file_type_config["csv_headers"] == True):
                    df=df.rename(columns = {column_names_relation[pair]:pair})
                else:
                    #df.columns.values[column_names_relation[pair]] = pair
                    df=df.rename(columns={ df.columns[column_names_relation[pair]]: pair })
            else:
                df=df.rename(columns = {column_names_relation[pair]:pair})
                
        # Extraer solo las columnas requeridas en el dataframe
        df = df[field_names]
        
        # Validar que no existan campos de user_id, item_id o rating vacios
        df = df.loc[pd.isnull(df.item_id) == False]
        df = df.loc[pd.isnull(df.user_id) == False]
        df = df.loc[pd.isnull(df.target) == False]

        #valida los campos vacios en todas las columnas y agregar un valor dependiendo del tipo de columna
        d1 = df.apply(lambda x: pd.api.types.infer_dtype(x.values))
        for column in field_names:
            if(d1[column] == "object"):
                df[column].fillna('e', inplace=True)
            else:
                df[column].fillna(0, inplace=True)
        
        # Validar los tipos de datos para item_id, user_id y rating
        # item_id
        if(("item_id" in field_names) and (d1["item_id"] == "float64")):
            return False,"Values on item_id can only be of type String or Int. Float values found."
        # user_id
        if(("user_id" in field_names) and (d1["user_id"] == "float64")):
            return False,"Values on user_id can only be of type String or Int. Float values found."
        # rating
        if(("target" in field_names) and (d1["target"] == "object")):
            return False,"Values on column 'target' can only be of type Int or Float. String values found."
        del(d1)
        
        # Validar que los valores del rating esten dentro del rango de valores definido
        # Remover los registros que tengan ratings con valores fuera del rango
        df = df.loc[df["target"].between(rating_config["min_rating"],rating_config["max_rating"]) == True]
        
        # Filtrar el contenido definido en configuración
        if(("remove_content" in fields) and (fields["remove_content"] != "")):
            remove_content = fields["remove_content"]
            for remove in remove_content:
                for remove_value in remove_content[remove]:
                    df = df[df[remove] != remove_value]

        return True, df


    def f(self, df_usa_polarity_desc):
        if df_usa_polarity_desc['sentiment'] > 0:
            val = 5
        elif df_usa_polarity_desc['sentiment'] == 0:
            val = 1
        else:
            val = 1
        return val



    # ------------------------------------
    # Function to get exlicit_text config and return dataframe
    def get_explicit_review_dataframe(self, config, df):
        interaction_required_fields = {
                    "user_id":False,
                    "item_id":False,
                    "review_text":False
                }
        
        fields = config['key_map']
        file_type_config = config['file_type_information']
        
        #Si se definio la columna rating en la configuración
        if(("rating" in fields) and ("rating_config" in fields) and (fields["rating_config"] != "")):
            # Obtene la configuración para la columna rating
            if(("rating_config" not in fields) or (fields["rating_config"] == "")):
                return False, "For explicit data is required to define rating_config property in configuration file"
            else:
                rating_config = fields["rating_config"]
                if(("min_rating" not in rating_config) or (rating_config["min_rating"] == None) or (type(rating_config["min_rating"]) is str)):
                    return False, "rating_config parameter in explicit configuration file must contain a 'min_rating' key with an int value"
                if(("max_rating" not in rating_config) or (rating_config["max_rating"] == None) or (type(rating_config["max_rating"]) is str)):
                    return False, "rating_config parameter in explicit configuration file must contain a 'max_rating' key with an int value"
                if(rating_config["min_rating"] >= rating_config["max_rating"]):
                    return False, "'max_rating' parameter must be greater and different from 'min_rating' parameter" 

        # Se obtienen las columnas declaradas en configuración
        field_names = []
        column_names_relation = {}
        for key in fields:
            if(fields[key] != ""):
                # Validar que los campos obligatorios esten definidos
                if(key in interaction_required_fields):
                    interaction_required_fields[key] = True
                if(key == "user_id"):
                    field_names.append("user_id")
                    column_names_relation["user_id"]=fields[key]
                if(key == "item_id"):
                    field_names.append("item_id")
                    column_names_relation["item_id"]=fields[key]
                if(key == "rating"):
                    field_names.append("rating")
                    column_names_relation["rating"]=fields[key]
                if(key == "review_title"):
                    field_names.append("review_title")
                    column_names_relation["review_title"]=fields[key]
                if(key == "review_text"):
                    field_names.append("review_text")
                    column_names_relation["review_text"]=fields[key]
                if(key == "timestamp"):
                    field_names.append("timestamp")
                    column_names_relation["timestamp"]=fields[key]
                if(key == "event_side_features"):
                    side_features = fields["event_side_features"]
                    for feature in side_features:
                        field_names.append(feature)
                        column_names_relation[feature]=side_features[feature]
                        
        # Se valida que todos los campos obligatorios esten definidos
        for valid_iter in interaction_required_fields:
            if(interaction_required_fields[valid_iter] == False):
                return False, "Required field is not defined in configuration. Field: "+str(valid_iter)
        
        # Cambiar/agregar nombres al dataframe
        for pair in column_names_relation:
            if(config["is_json"] == False):
                if(file_type_config["csv_headers"] == True):
                    df=df.rename(columns = {column_names_relation[pair]:pair})
                else:
                    df.columns.values[column_names_relation[pair]] = pair
            else:
                df=df.rename(columns = {column_names_relation[pair]:pair})
                
        # Extraer solo las columnas requeridas en el dataframe
        df = df[field_names]
        
        # Validar que no existan campos vacios en user_id, item_id y review_text
        df = df.loc[pd.isnull(df.item_id) == False]
        df = df.loc[pd.isnull(df.user_id) == False]
        df = df.loc[pd.isnull(df.review_text) == False]

        #valida los campos vacios en todas las columnas y agregar un valor dependiendo del tipo de columna
        d1 = df.apply(lambda x: pd.api.types.infer_dtype(x.values))
        
        for column in field_names:
            if(d1[column] == "object"):
                df[column].fillna('e', inplace=True)
            else:
                df[column].fillna(0, inplace=True)
        
        # Validar los tipos de datos para item_id, user_id, review_title y review_text
        # item_id
        d1 = df.apply(lambda x: pd.api.types.infer_dtype(x.values))
        
        if(("item_id" in field_names) and (d1["item_id"] == "float64")):
            return False,"Values on item_id can only be of type String or Int. Float values found."
        # user_id
        if(("user_id" in field_names) and (d1["user_id"] == "float64")):
            return False,"Values on user_id can only be of type String or Int. Float values found."
        # review_title
        if(("review_title" in field_names) and (d1["review_title"] not in ["object","unicode"])):
            return False,"Values on column 'review_title' can only be of type String. Int or Float values found."
        # review_text
        if(("review_text" in field_names) and (d1["review_text"] not in ["object","unicode"])):
            return False,"Values on column 'review_text' can only be of type String. Int or Float values found."
        del(d1)
        
        # Obtener un valor numerico o rating desde el texto del comentario haciendo uso de analisis de sentimientos
        # If there is a review_title column
        if("review_title" in field_names):
            # Append review_title and review_text columns in full_review_text column
            df['full_review_text'] = df["review_title"]+ " " + df["review_text"].map(str)
        else:
            df['full_review_text'] = df["review_text"].map(str)
        
        # Validar que no existan campos vacios en full_review_text
        df = df.loc[pd.isnull(df.full_review_text) == False]
        
        # Se remueven las columnas 'review_title' y 'review_text' del dataframe
        if('review_title' in field_names):
            df = df.drop(['review_title', 'review_text'], axis=1)
        else:
            df = df.drop(['review_text'], axis=1)
        
        # Se halla la polaridad de sentimiento en cada texto de review
        st = default_timer()
        bloblist_desc = list()

        df_usa_descr_str=df['full_review_text'].astype(str)
        print("Total number of texts to transform: "+str(len(df_usa_descr_str)))
        word_count = 0
        for row in df_usa_descr_str:
            word_count += 1
            print("Transformed texts: "+str(word_count))
            blob = TextBlob(row)
            bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
            df_usa_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])

        df['target'] = df_usa_polarity_desc.apply(self.f, axis=1)

        # Se remueve la columna 'full_review_text'
        df = df.drop(['full_review_text'], axis=1)

        runtime = default_timer() - st
        # Almacenar tiempo de proceso en base de datos
        new_entry= pd.DataFrame(columns=['event','description','process_time','datetime'])
        new_entry = new_entry.append({'event':"get_polarity_from_review_text",'description': "Tiempo que toma obtener la polaridad de todos los review de la fuente de datos explicit_review","process_time":round(runtime,2), "datetime":datetime.datetime.now()}, ignore_index=True)
        new_entry.to_sql("time_results", sql_db, if_exists='append')
        del(new_entry)

        #Si se definio la columna rating en la configuración
        if(("rating" in fields) and (fields["rating_config"] != "")):
            # Validar que los valores del rating esten dentro del rango de valores definido
            # Remover los registros que tengan ratings con valores fuera del rango
            df = df.loc[df["rating"].between(rating_config["min_rating"],rating_config["max_rating"]) == True]
        
        # Filtrar el contenido definido en configuración
        if(("remove_content" in fields) and (fields["remove_content"] != "")):
            remove_content = fields["remove_content"]
            for remove in remove_content:
                for remove_value in remove_content[remove]:
                    df = df[df[remove] != remove_value]

        return True, df



    # ------------------------------------
    # Function to get item_content config and return dataframe
    def get_item_content_dataframe(self, config, df):
        interaction_required_fields = {
                    "item_id":False,
                }
        
        fields = config['key_map']
        file_type_config = config['file_type_information']
        
        # Se obtienen las columnas declaradas en configuración
        field_names = []
        column_names_relation = {}
        for key in fields:
            if(fields[key] != ""):
                # Validar que los campos obligatorios esten definidos
                if(key in interaction_required_fields):
                    interaction_required_fields[key] = True
                if(key == "item_id"):
                    field_names.append("item_id")
                    column_names_relation["item_id"]=fields[key]
                if(key == "item_description"):
                    side_features = fields["item_description"]
                    for feature in side_features:
                        field_names.append(feature)
                        column_names_relation[feature]=side_features[feature]
                if(key == "other_features"):
                    side_features = fields["other_features"]
                    for feature in side_features:
                        field_names.append(feature)
                        column_names_relation[feature]=side_features[feature]
                        
        # Se valida que todos los campos obligatorios esten definidos
        for valid_iter in interaction_required_fields:
            if(interaction_required_fields[valid_iter] == False):
                return False, "Required field is not defined in configuration. Field: "+str(valid_iter)
        
        # Cambiar/agregar nombres al dataframe
        for pair in column_names_relation:
            if(config["is_json"] == False):
                if(file_type_config["csv_headers"] == True):
                    df=df.rename(columns = {column_names_relation[pair]:pair})
                else:
                    df.columns.values[column_names_relation[pair]] = pair
            else:
                df=df.rename(columns = {column_names_relation[pair]:pair})
                
        # Extraer solo las columnas requeridas en el dataframe
        df = df[field_names]
        
        # Validar que no existan campos vacios en item_id
        df = df.loc[pd.isnull(df.item_id) == False]

        #valida los campos vacios en todas las columnas y agregar un valor dependiendo del tipo de columna
        d1 = df.apply(lambda x: pd.api.types.infer_dtype(x.values))
        for column in field_names:
            if(d1[column] == "object"):
                df[column].fillna('e', inplace=True)
            else:
                df[column].fillna(0, inplace=True)
        
        # Validar los tipos de datos para item_id
        # item_id
        if(d1["item_id"] == "float64"):
            return False,"Values on item_id can only be of type String or Int. Float values found."
        del(d1)
        
        # Filtrar el contenido definido en configuración
        if(("remove_content" in fields) and (fields["remove_content"] != "")):
            remove_content = fields["remove_content"]
            for remove in remove_content:
                for remove_value in remove_content[remove]:
                    df = df[df[remove] != remove_value]

        return True, df



    # ------------------------------------
    # Function to get user_content config and return dataframe
    def get_user_content_dataframe(self, config, df):
        interaction_required_fields = {
                    "user_id":False,
                }
        
        fields = config['key_map']
        file_type_config = config['file_type_information']
        
        # Se obtienen las columnas declaradas en configuración
        field_names = []
        column_names_relation = {}
        for key in fields:
            if(fields[key] != ""):
                # Validar que los campos obligatorios esten definidos
                if(key in interaction_required_fields):
                    interaction_required_fields[key] = True
                if(key == "user_id"):
                    field_names.append("user_id")
                    column_names_relation["user_id"]=fields[key]
                if(key == "user_description"):
                    side_features = fields["user_description"]
                    for feature in side_features:
                        field_names.append(feature)
                        column_names_relation[feature]=side_features[feature]
                if(key == "other_features"):
                    side_features = fields["other_features"]
                    for feature in side_features:
                        field_names.append(feature)
                        column_names_relation[feature]=side_features[feature]
                        
        # Se valida que todos los campos obligatorios esten definidos
        for valid_iter in interaction_required_fields:
            if(interaction_required_fields[valid_iter] == False):
                return False, "Required field is not defined in configuration. Field: "+str(valid_iter)
    
        
        # Cambiar/agregar nombres al dataframe
        for pair in column_names_relation:
            if(config["is_json"] == False):
                if(file_type_config["csv_headers"] == True):
                    df=df.rename(columns = {column_names_relation[pair]:pair})
                else:
                    df.columns.values[column_names_relation[pair]] = pair
            else:
                df=df.rename(columns = {column_names_relation[pair]:pair})
                
        # Extraer solo las columnas requeridas en el dataframe
        df = df[field_names] 

        # Validar que no existan campos vacios en user_id
        df = df.loc[pd.isnull(df.user_id) == False]

        #valida los campos vacios en todas las columnas y agregar un valor dependiendo del tipo de columna
        d1 = df.apply(lambda x: pd.api.types.infer_dtype(x.values))
        for column in field_names:
            if(d1[column] == "object"):
                df[column].fillna('e', inplace=True)
            else:
                df[column].fillna(0, inplace=True)
        
        # Validar los tipos de datos para item_id
        # item_id
        if(d1["user_id"] == "float64"):
            return False,"Values on item_id can only be of type String or Int. Float values found."
        del(d1)

        # Filtrar el contenido definido en configuración
        if(("remove_content" in fields) and (fields["remove_content"] != "")):
            remove_content = fields["remove_content"]
            for remove in remove_content:
                for remove_value in remove_content[remove]:
                    df = df[df[remove] != remove_value]

        return True, df


    




