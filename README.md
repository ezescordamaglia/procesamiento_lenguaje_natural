# CEIA - UBA - Procesamiento de Lenguaje Natural

En este repositorio se han realizado los desafíos referidos a la materia de Procesamiento del Lenguaje Natural de la Especializacion en Inteligencia Artificial.

# Desafío 1 - Vectorización y Similaridad entre documentos:
En este desafío se programan manualmente las funciones para obtener el vocabulario de un corpus y vectorizar por OneHot encoding, Vector de frecuencia y TF-IDF. 
Tambien se programa una función para comparar similaridad coseno entre todos los documentos del corpus segun distintas formas de vectorización.

[Link al Desafío 1](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_1/1a%20-%20vectorizacion.ipynb)

# Desafío 2 - Busqueda en Corpus y Boy de preguntas y respuestas predeterminadas:
En este desafío se toma como corpus un articulo de Wikipedia que habla sobre el tren (https://es.wikipedia.org/wiki/Tren), se limpia y preprocesa y se utilizan librerias de NLTK con un Steamer en español para tokenizar los documentos.
Luego se programa una funcion que recibe una pregunta del usuario y busca en el corpus el documento que tiene mas similaridad coseno con el texto ingresado (utilizando vectorización TF-IDF) y lo devuelve.

[Link al Desafío 2 - NLTK](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_2/2c%20-%20bot_tfidf_nltk.ipynb)

Tambien se construyó un bot de consultas abiertas y preguntas predeterminadas para un banco utilizando las librerias de Spicy y Stranza. 
Se generó un diccionario de entradas con "tags", como puden ser binvenida, contacto y productos, cuyos tags contenian patrones, o consultas ingresadas por el usuario y respuestas fijas para esas consultas.
La idea del desafío fue vectorizar ese diccionario con TF-IDF y entrenar un modelo de Deep Learning que aprenda a devolver una respuesta preesablecida segun lo ingresado por el usuario.
Al comparar lo ingresado por el usuario contra los patrones esperados, las respuestas suelen ser mas acertadas que el anterior, pero suele tener fallos cuando lo ingresado por el usuario no se parece a ningun patron esperado.

[Link al Desafío 2 - BOT DE CONSULTAS](https://github.com/ezescordamaglia/procesamiento_lenguaje_natural/blob/main/Desafio_2/2b%20-%20bot_dnn_spacy_esp.ipynb)
