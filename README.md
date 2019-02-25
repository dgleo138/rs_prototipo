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
1. **python_files:** Contiene todos los archivos con código en Python.
2. **Datasets:** Contiene los directorios de los proveedores o clientes del sistema.
  - *Nombre_cliente:* Permite almacenar los datos de un cliente específico
      - *data:* Directorio que contiene los archivos de configuración y los directorios que almacenan los archivos con datos para diferentes fuentes de información.
        - *fuente_información:* (ejemplo: explicit) Directorio que contiene todos los archivos con datos que serán usados como entrada para los algoritmos de recomendación para la fuente de información especifica. Los archivos deben tener el mismo formato (csv, json) y estructura, los cuales deben corresponder en lo definido en el archivo de configuración para la fuente específica.
        - *configuración_fuente_datos:* (ejemplo: explicit_conf.json) En este archivo se definen propiedades de los archivos de datos a usar por la fuente específica. Acá se define el tipo y formato de archivo, el mapeo de la estructura de datos original a la estructura usada por los algoritmos y algunas opciones de limpieza de datos. 
      - *data_backup:* Directorio que almacena los archivos de datos procesados para las diferentes fuentes de información
      - *database:* Directorio que contiene la base de datos del cliente, los modelos entrenados y las predicciones para hibridaciones.
      - *config.ini:* Archivo de configuración que contiene todas las propiedades generales de ejecución para el cliente.

## Conjuntos de datos
Los conjuntos de datos usados en la validación del prototipo se comparten en el siguiente enlace: https://eafit-my.sharepoint.com/:f:/g/personal/dgonza37_eafit_edu_co/ErHzJslkaENMsfr7ai6ypdkBMx0S3OXapYC9lEuEu88GUg?e=QwChcM


## Funciones y paso a paso:

Este prototipo no incluye una capa de servicios desde la cual se puedan ejecutar los procesos y consumir las recomendaciones por parte de clientes y proveedores. Por lo tanto, es necesario ejecutar cada función desde la consola de comandos, usando python.

Para usar las funciones del sistema, es necesario usar la consola de comandos y ubicarse sobre el directorio *python_files*. Luego se pueden usar los siguientes comandos para ejecutar cada uno de los procesos:

1. Estandarizar los datos almacenados en el directorio de un cliente específico. Se debe cambiar *nombre_del_directorio_del_cliente* por el nombre del cliente o directorio para el que se quiere ejecutar el proceso. Por ejemplo: Amazon.

Comando: **python -c 'import provider_services; provider_services.data_sources_etl("*nombre_del_directorio_del_cliente*")'**

2. Priorizar los algoritmos haciendo uso de los datos estandarizados de un cliente. Es necesario haber ejecutado previamente el proceso de estandarización de datos del cliente para el que se van a priorizar los algoritmos. Se debe cambiar *nombre_del_directorio_del_cliente* por el nombre del cliente o directorio para el que se quiere ejecutar el proceso. Por ejemplo: Amazon.

Comando: **python -c 'import provider_services; provider_services.algorithm_priorization("*nombre_del_directorio_del_cliente*")'**



Dider León González Arroyave

Correo: dgonza37@eafit.edu.co
