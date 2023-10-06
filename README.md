# CEIA - UBA - Procesamiento de Lenguaje Natural
Año 2023

Docente: Rodrigo Cárdenas

Alumno: Scordamaglia Ezequiel

## Presentación

En este repositorio se han realizado los desafíos referidos a la materia de Procesamiento del Lenguaje Natural de la Especialización en Inteligencia Artificial.



## Conceptos aplicados
- Vectorización: Frecuencias, One Hot Encoding, TF-IDF
- Similaridad Coseno
- Steamer / Lematizador
- Tokenización
  

## Librerías utilizadas
- numpy
- nltk
- gradio
- sklearn
- stanza
- spacy_stanza
- tensorflow
- keras
- seaborn
- matplotlib

## Lenguaje

Python

## Desafío 1 - Vectorización y Similaridad entre documentos:
<img src="https://old.tacosdedatos.com/assets/detrasdelavis/004_tfidf.png" width="600" height="150">
En este desafío se programan manualmente las funciones para obtener el vocabulario de un corpus y vectorizar por OneHot encoding, Vector de frecuencia y TF-IDF. 
También se programa una función para comparar similaridad coseno entre todos los documentos del corpus segun distintas formas de vectorización.

[Link al Desafío 1](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_1/1a%20-%20vectorizacion.ipynb)

## Desafío 2 - Búsqueda en Corpus y Bot de preguntas y respuestas predeterminadas:

<img src="https://img.freepik.com/vector-premium/libro-icono-lupa-ilustracion-vectorial_676691-1294.jpg" width="600" height="300">

En este desafío se toma como corpus un artículo de Wikipedia que habla sobre el tren (https://es.wikipedia.org/wiki/Tren), se limpia y preprocesa, y se utilizan librerías de NLTK con un Steamer en español para tokenizar los documentos.
Luego se programa una función que recibe una pregunta del usuario y busca en el corpus el documento que tiene mas similaridad coseno con el texto ingresado (utilizando vectorización TF-IDF) y lo devuelve.

[Link al Desafío 2 - NLTK](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_2/2c%20-%20bot_tfidf_nltk.ipynb)

Tambien se construyó un Bot de consultas abiertas y preguntas predeterminadas para una entidad bancaria utilizando las librerías de Spicy y Stranza. 
Se generó un diccionario de entradas con "tags", como "binvenida", "contacto" y "productos", los cuales contenían patrones, o consultas ingresadas por el usuario, y respuestas fijas para esas consultas.
La idea del desafío fue vectorizar ese diccionario con TF-IDF y entrenar un modelo de Deep Learning que aprenda a devolver una respuesta preesablecida según lo ingresado por el usuario.
Al comparar lo ingresado por el usuario contra los patrones esperados, las respuestas suelen ser mas acertadas que el notebook anterior, pero suele tener fallos cuando lo ingresado por el usuario no se parece a ningún patron esperado.

[Link al Desafío 2 - BOT DE CONSULTAS](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_2/2b%20-%20bot_dnn_spacy_esp.ipynb)

## Desafío 3 - Embeddings:

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSB4ufYX8qZoEdx_f79fLhT5Sgb6sodqm9aEQ&usqp=CAU" width="500" height="150">

El objetivo de este desafío fue utilizar un corpus base para crear embeddings de palabras basados en ese contexto.
En este caso se utilizó el libro "El origen de las especies" de Charles Darwin en formato .txt (https://www.textos.info/charles-darwin/el-origen-de-las-especies), se preprocesó y se utilizó la libreria Gensim (Word2Vec) para entrenar el modelo generador de Embeddings. 
Se consiguieron buenos resultados en la generación de Embeddings en el contexto del libro y se hicieron distintas pruebas y tests de analogías muy interesantes.
Algunos de los resultados mas sorprendentes fueron: 

**SEMILLA + TIERRA = HOJA**

**ANIMALES + FRIO = MORIR**

**TIERRA - AGUA = DESIERTO**

[Link al Desafío 3 - EMBEDDINGS](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_3/3b%20-%20Custom%20embedding%20con%20Gensim.ipynb)

## Desafío 4 - Predicción de próximas palabras:
<img src="https://editor.analyticsvidhya.com/uploads/782781__MrDp6w3Xc-yLuCTbco0xw.png" width="500" height="150">
El objetivo de este desafío era predecir la próxima palabra dado un conjunto de palabras iniciales. Para este ejercicio se utiliza como dataset un conjunto de fragmentos de libros de Borges, Cortazar y Bioy Casares. Los mismos se preprocesaron y formatearon en secuencias de 4 palabras: las primeras 3 utilizadas como referencia y la última era la palabra a predecir.
Con ese set de datos se entrenó un modelo de Red Neuronal Recurrente para que devuelva un vector softmax con la probabilidad de cada una de las palabras del vocabulario.
Si bien el modelo no logró un buen accuracy en validación, se probó el modelo para generar nuevas palabras a partir de las ingresadas por el usuario, con algunos resultados interesantes:


**INPUT = 'Estaba mirando que'**

**OUTPUT = 'Estaba mirando que un poco la'**

[Link al Desafío 4 - PROXIMA PALABRA](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_4/4d%20-%20predicci%C3%B3n_palabra.ipynb)

Tambien se intentó entrenar el mismo modelo, pero utilizando embeddings pre-entrenados de Word2Vec, logrando un accuracy y un resultado en las pruebas levemente mejor.

[Link al Desafío 4 - PROXIMA PALABRA + WORD2VEC](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_4/4d%20-%20predicci%C3%B3n_palabra_wordtovec.ipynb)

## Desafío 5 - Análisis de Sentimientos:
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSBLDeyiZMAhccYz4ATcAKo-z0Azbm9QZyhoA&usqp=CAU" width="500" height="150">

En este desafío se analizaron críticas de compradores de ropa para clasificar el comentario del 1 al 5 en cuanto a la satisfacción del cliente.
Se utilizó el dataset "clothing_ecommerce_reviews.csv" (https://drive.google.com/uc?id=1Urn1UFSrodN5BuW6-sc_igtaySGRwhV8), se limpió, se preprocesó y se tokenizó, para armar secuencias de entrada a un modelo de Red Neuronal Recurrente.
Dentro de cada notebook se comparan modelos que entrenan sus propios embeddings contra modelos que utilizan embeddings pre-entrenados por Fasttext.

En el primer notebook se utilizan los datos tal cual fueron recibidos, con mucho desbalance y separados en 5 niveles de satisfacción, obteniendo un resultado no muy bueno:

[Link al Desafío 5 - CLASIFICACION DE SENTIMIENTOS DE 1 A 5](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_5/5%20-%20clothing_ecommerce_reviews.ipynb)

En el segundo notebook se combinan los niveles de satisfacción para dejar solo 3 y se utilizan técnicas de balanceo del dataset (Oversampling y Undersampling), logrando un mejor accuracy del modelo.

[Link al Desafío 5 - CLASIFICACION DE SENTIMIENTOS DE 1 A 3 - BALANCEADO](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_5/5%20-%20clothing_ecommerce_reviews_balanced_unified.ipynb)

En el tercer notebook se combinan los niveles de satisfacción para dejar solo 3 y se utiliza una librería llamada NLPAUG para generar nuevas reseñas basadas en reseñas existentes y alterando algunas palabras por sinónimos. Este último modelo es que que ubtuvo un mejor accuracy en validación y mejores resultados en las pruebas.

[Link al Desafío 5 - CLASIFICACION DE SENTIMIENTOS DE 1 A 3 - BALANCEADO - SINONIMOS](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_5/5%20-%20clothing_ecommerce_reviews_synonym.ipynb)

## Desafío 6 - Bot Conversacional:

<img src="https://blog.tactium.com.br/wp-content/themes/softium/script/timthumb.php?src=https://blog.tactium.com.br/wp-content/uploads/2022/07/chatbots-robo-de-voz.jpg&w=930&h=480&zc=1&q=100" width="500" height="200">

En este desafío se construye un Bot conversacional que responde preguntas del usuario. Se utilizan como base los datos del challenge ConvAI2 (Conversational Intelligence Challenge 2) de conversaciones en inglés (http://convai.io/data/). Se preprocesa y se limpian los datos, y se entrena un modelo Encoder-Decoder que utiliza los embeddigns pre-entrenados de Glove, logrando un accuracy de 80.7%

![alt text](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_6/model_plot.png?raw=true)

[Link al Desafío 6 - BOT CONVERSACIONAL](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_6/6-%20bot_qa.ipynb)


