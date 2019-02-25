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
  - *Nombre_cliente:* Permite almacenar los datos de un cliente específico
      - *data:* Directorio que contiene los archivos de configuración y los directorios que almacenan los archivos con datos para diferentes fuentes de información.
        - fuente_información: (ejemplo: explicit) Directorio que contiene todos los archivos con datos que serán usados como entrada para los algoritmos de recomendación para la fuente de información especifica. Los archivos deben tener el mismo formato (csv, json) y estructura, los cuales deben corresponder en lo definido en el archivo de configuración para la fuente específica.
        - configuración_fuente_datos: (ejemplo: explicit_conf.json) En este archivo se definen propiedades de los archivos de datos a usar por la fuente específica. Acá se define el tipo y formato de archivo, el mapeo de la estructura de datos original a la estructura usada por los algoritmos y algunas opciones de limpieza de datos. 
      - *data_backup:* Directorio que almacena los archivos de datos procesados para las diferentes fuentes de información
      - *database:* Directorio que contiene la base de datos del cliente, los modelos entrenados y las predicciones para hibridaciones.
      - *config.ini:* Archivo de configuración que contiene todas las propiedades generales de ejecución para el cliente.

## Conjuntos de datos


## Funciones y paso a paso:

Este prototipo no incluye una capa de servicios desde la cual se puedan ejecutar los procesos y consumir las recomendaciones por parte de clientes y proveedores. Por lo tanto, es necesario ejecutar cada función desde la consola de comandos, usando python.

Para usar las funciones del sistema, se debe usar el siguiente comando en la linea de comandos:  
**python -c 'import provider_services; provider_services.*nombre_del_proceso_a_ejecutar*("*nombre_del_directorio_del_cliente*")'**

El atributo ***nombre_del_directorio_del_cliente*** corresponde al nombre que se da al directorio en el que se almacenan los datos del cliente, incluidos los conjuntos de datos que seran usados por el sistema.

El método ***nombre_del_proceso_a_ejecutar*** corresponde a la acción que se requiere realizar por el sistema. Los métodos permitidos son:

python -c 'import provider_services; provider_services.data_sources_etl("cliente_test")'
python -c 'import provider_services; provider_services.algorithm_priorization("cliente_test")'
