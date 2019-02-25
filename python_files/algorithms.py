#!~/dider/bin/python
# -*- coding: UTF-8 -*-

# Import local scripts
import common

# Recommender system Algorithms

# Popularity Model

def custom_popularity(df):
    # Convertir df a dataframe
    try:
        # Se agrupan todos los eventos que ha tenido un usuario con un item y se suman los valores de target para esa relación
        df = df \
                        .groupby(['user_id', 'item_id'])['target'].sum() \
                        .apply(common.smooth_user_preference).reset_index()

        # Se suman los valores de target en todos los usuarios para cada item 
        df = df.groupby('item_id')['target'].sum().sort_values(ascending=False).reset_index()
        #Generate a recommendation rank based upon score
        df['Rank'] = df['target'].rank(ascending=0, method='first')
        #print(df.head(10))

        return {"status":True, "result":df}
    except ValueError as e:
        return {"status":False, "result":"There was an error trying to compute custom_popularity. Error: "+str(e)}


