# Define some sevices from which provider can consume every process used in this prototype

#class prototypeServices:

 #   def __init__(self, client_name):
        # Nombre del cliente para identificar sobre cuales archivos se ejecutaran los procesos
  #      self.client_name = name

from main import main

# Estandarizar los datos de los archivos almacenados para un proveedor especifico
def data_sources_etl(client):
    base_class = main(client)
    base_class.get_source_data() 

# Estandarizar los datos de los archivos almacenados para un proveedor especifico
def algorithm_priorization(client):
    base_class = main(client)
    base_class.run_and_evaluate_algorithms() 