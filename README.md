# rs_prototipo
Prototipo para validar el diseño de sistema de recomendación propuesto en el trabajo de grado de la maestría en profundización.

Este prototipo se usa para validar un sistema de recomendación que estandariza los conjuntos de datos y presenta un mecanismo que explora diferentes algoritmos, prioriza y selecciona el que mejor lo hace según un conjunto de métricas definidas.

## Dependencias:
 - Python versión 3.7.0
 - pandas: https://pandas.pydata.org/pandas-docs/stable/install.html
 - sqlalchemy: https://pypi.org/project/SQLAlchemy/
 - surprise: http://surpriselib.com/
 - numpy: https://scipy.org/install.html
 - configparser: https://pypi.org/project/configparser/
 
## Directorios y Archivos:
1. python_files: Contiene todos los archivos con código en Python.
2. Datasets: Contiene los directorios de los proveedores o clientes del sistema.
 2.1. *Nombre_cliente:*
  2.1.1. *data:* Contiene los archivos de configuración y los directorios que almacenan los archivos con datos para diferentes fuentes de información.
  2.1.2. *data_backup:* Almacena los archivos de datos procesados para las diferentes fuentes de información
  2.1.3. *database:* Contiene la base de datos del cliente, los modelos entrenados y las predicciones para hibridaciones.

## Conjuntos de datos


## Funciones y paso a paso:

Este prototipo no incluye una capa de servicios desde la cual se puedan ejecutar los procesos y consumir las recomendaciones por parte de clientes y proveedores. Por lo tanto, es necesario ejecutar cada función desde la consola de comandos, usando python.

Para usar las funciones del sistema, se debe usar el siguiente comando en la linea de comandos:  
**python -c 'import provider_services; provider_services.*nombre_del_proceso_a_ejecutar*("*nombre_del_directorio_del_cliente*")'**

El atributo ***nombre_del_directorio_del_cliente*** corresponde al nombre que se da al directorio en el que se almacenan los datos del cliente, incluidos los conjuntos de datos que seran usados por el sistema.

El método ***nombre_del_proceso_a_ejecutar*** corresponde a la acción que se requiere realizar por el sistema. Los métodos permitidos son:

python -c 'import provider_services; provider_services.data_sources_etl("cliente_test")'
python -c 'import provider_services; provider_services.algorithm_priorization("cliente_test")'
