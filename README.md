# Mercado_inmobiliario_Machine_Learning

## _Descripción:_
Se realizó un modelo de machine learning para realizar una predicción sobre si una casa es barata o cara.

## _Tecnologías usadas:_
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![RailWay](https://img.shields.io/badge/Railway-131415?style=for-the-badge&logo=railway&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## _Librerías:_
<ul>
    <li><strong>pandas</strong>: Es una herramienta de análisis y manipulación de datos de código abierto rápida, potente, flexible y fácil de usar, construido sobre el lenguaje de programación Python.</li>
    <li><strong>numpy</strong>: Librería de Python especializada en el cálculo numérico y el análisis de datos, especialmente para un gran volumen de datos.</li>
    <li><strong>re</strong>: Este módulo proporciona operaciones de coincidencia de expresiones regulares similares a las encontradas en Perl.</li>
    <li><strong>folium</strong>: Es una librería de Python que se usa para visualizar mapas</li>
    <li><strong>geopy</strong>: geopy facilita que los desarrolladores de Python ubiquen las coordenadas de direcciones, ciudades, países y puntos de referencia en todo el mundo utilizando geocodificadores de terceros y otras fuentes de datos.</li>
    <li><strong>matplotlib</strong>: Es una librería de Python especializada en la creación de gráficos en dos dimensiones.</li>
    <li><strong>seaborn</strong>: Es una librería de visualización de datos.</li>
    <li><strong>tqdm</strong>: Es un pequeño módulo que permite crear una barra de progreso basada en texto, que es desplegada en pantalla a partir de un bucle</li>
    <li><strong>scikit-learn</strong>: También llamada sklearn, es un conjunto de rutinas escritas en Python para hacer análisis predictivo, que incluyen clasificadores, algoritmos de clusterización, etc. Está basada en NumPy, SciPy y matplotlib.</li>
    <li><strong>tensorflow</strong>: Es una biblioteca de código abierto para aprendizaje automático a través de un rango de tareas, y desarrollado por Google para satisfacer sus necesidades de sistemas capaces de construir y entrenar redes neuronales para detectar y descifrar patrones y correlaciones, análogos al aprendizaje y razonamiento usados por los humanos.</li>
    <li><strong>joblib</strong>: Es un conjunto de herramientas para proporcionar un Pipeline ligero en Python.</li>
    <li><strong>streamlit</strong>: Es una librería que, de forma sencilla, te permite crear todo tipo de aplicaciones de datos desarrolladas en Python</li>
</ul>

## _Objetivo del proyecto:_
Usted ha sido contactado de una importante empresa inversora dentro del rubro de la inmobiliaria en Colombia, con el fin de que implemente un modelo de clasificación que permita clasificar el precio de las propiedades en venta, utilizando los datos que se han puesto a su disposición correspondientes al año 2020. Para esto, específicamente, debe predecir la categorización de las propiedades entre baratas o caras, considerando como criterio el valor promedio de los precios.

## _Resumen de la solución del proyecto:_
<ul>
  <li>
    <strong>Cargar el dataset y transformalo en DataFrames.</strong>
    <a href="https://drive.google.com/file/d/1AQG7XUFqT6D6WM_WK--j1IcZkPF3LOLZ/view?usp=sharing">(descargar dataset)</a>
  </li>
  <li><strong>Limpieza, análisis exploratorio y transformación de los DataFrames.</strong></li>
  <li><strong>Codificación de variables categóricas.</strong></li>
  <li><strong>Extracción de las variables numéricas.</strong></li>
  <li><strong>Vector objetivo.</strong></li>
  <li><strong>Escalado de datos para no tener sesgos hacia una variable debido a su tamaño con respecto a otra.</strong></li>
  <li><strong>Crear un set de entrenamiento y de prueba.</strong></li>
  <li><strong>Creación y evaluación de las métricas del modelo de Regresión Logística.</strong></li>
  <li><strong>Creación y evaluación de las métricas del modelo de Bosques aleatorios.</strong></li>
  <li><strong>Creación y evaluación de las métricas del modelo de Redes Neuronales Artificiales.</strong></li>
  <li><strong>Comparación de los 3 modelos.</strong></li>
  <li>
    <strong>Creación del archivo del mejor modelo.</strong>
    <a href="https://drive.google.com/file/d/1XXkZ2BpaxXJcdrdr_Zw04BL5HeP_LYEi/view?usp=sharing">(descargar modelo)</a>
  </li>
  <li><strong>Diseño de la aplicación web usando streamlit.</strong></li>
</ul>

## _Proceso detallado de la solución del proyecto:_
El proceso detallado desde la carga del dataset hasta la puesta en producción del modelo de Machine Learning se describen a continuación:

<h3>
  <h4>
    <ul>
      <li>
        <strong>Extracción, transformación, limpieza y creación del archivo del modelo de machine learning: </strong>
        <a href="https://github.com/JozCrzBrgn/Mercado_inmobiliario_Machine_Learning/blob/main/Mercado_Inmobiliario.ipynb">Jupyter Notebook</a>
      </li>
    </ul>
   <h4>
</h3>
    
## _Aplicación web:_

<h3>
  <h4>
    <ul>
      <li>
        <strong>Mercado Inmobiliario</strong>
        <a href="https://huggingface.co/spaces/JozCrzBrgn/Mercado_Inmobiliario">Demo de la App-Web</a>
      </li>
    </ul>
   <h4>
</h3>
<img src="https://github.com/JozCrzBrgn/Mercado_inmobiliario_Machine_Learning/blob/main/_img/web.png">
