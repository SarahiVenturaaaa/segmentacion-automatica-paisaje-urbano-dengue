<table width="130%">
  <tr>
    <td align="left" width="25%">
      <img src="logocimat.png" alt="Logo CIMAT" width="120">
    </td>
    <td align="center" width="50%">
      <h2>Centro de Investigación en Matemáticas, A.C.</h2>
    </td>
    <td align="right" width="35%">
      <img src="logo_secihti.png" alt="Logo SECiHTI" width="450">
    </td>
  </tr>
</table>


<p align="center">
  <img src="logo_insp.png" alt="Logo INSP" width="250"/>
</p>
<br>

<h1 align="center">Segmentación semántica del paisaje en zonas urbanas para la determinación de factores de riesgo de transmisión de dengue</h1>


Este repositorio contiene el conjunto de datos, los notebooks, las herramientas y la documentación generados durante un proyecto realizado en colaboración con el Instituto Nacional de Salud Pública (INSP).

Se desarrolló una herramienta de segmentación semántica del paisaje urbano en Tapachula, Chiapas, utilizando un enfonque basado en **superpíxeles (SLIC)** y **modelos de aprendizaje automático**, con el objetivo de identificar de forma automática distintas coberturas del paisaje, como vegetación, cuerpos de agua, techos, pavimento, entre otras, que están asociadas a factores de riesgo para la transmisión de enfermedades como el dengue (mosquito *Aedes aegypti*).

## 📂 Contenido del repositorio

- **Herramienta de etiquetado** para la generación del conjunto de datos:
  - Segmentación de la imagen en superpíxeles (SLIC).
  - Extracción de **129 características por superpíxel**: estadísticas de color (RGB), bordes/estructura y textura (GLCM).
  - Generación de **2 archivos por imagen**:
    - `.csv` con el vector de características por superpíxel y la etiqueta asignada,
    - `.tif` con la imagen segmentada y etiquetada (visualización).


 - **Modelo de aprendizaje automático seleccionado:**

Se evaluaron diferentes modelos de aprendizaje automático (KNN, SVM y MLP) para la clasificación de las coberturas del paisaje, y el modelo <strong>Multilayer Perceptron (MLP)</strong> presentó el mejor desempeño global weighted sobre el conjunto de prueba, con una *precisión* de *0.8199*, un *recall* de *0.8193* y un *F1-score* de *0.8086*. Por esta razón, fue seleccionado como el modelo final para la clasificación automática de las coberturas del paisaje.

- **Clasificación automática de imágenes**:
  - Generación de resultados de imágenes **procesadas y clasificadas automáticamente** (10 coberturas + “Sin etiqueta”), utilizando los modelos entrenados y la estrategia seleccionada.
  - Salidas por imagen:
  - Imagen segmentada y clasificada, con leyenda que muestra las proporciones de superpíxeles por categoría.
  - Archivo `.csv` con el vector de características por superpíxel (y la predicción/etiqueta correspondiente) para cada imagen procesada.
  - **Visualizaciones adicionales**:
    - superpíxeles por categoría,
    - gráficas de proporción por clase (barras y pastel) para cada imagen procesada.
 
***

## 🛠 Requisitos para la generación del conjunto de datos

Para generar el conjunto de datos, es necesario ejecutar el script de la herramienta de etiquetado **`Etiquetador_actualizado.py`**. Con el fin de garantizar su correcta ejecución, se debe crear un **entorno de Python** con la versión **Python 3.9.18** e instalar las librerías necesarias de acuerdo con las versiones señaladas. Estas especificaciones se encuentran en el archivo **`requerimientos_etiquetador.txt`**.

**Nota:** El entorno puede crearse con la herramienta de preferencia, por ejemplo, `venv`, `virtualenv` o Conda. Lo importante es conservar la versión de Python indicada y respetar las dependencias especificadas en el archivo de requerimientos.

Una vez creado y activado el entorno, se puede consultar el video **`tutorial_herramienta_etiquetado_RGB`**, en el cual se muestra el procedimiento paso a paso para utilizar la herramienta de etiquetado manual. El video se encuentra disponible en:

https://drive.google.com/file/d/1YwgJuyjX_4MJxT2aH8uJotcvUu8PKUlF/view?usp=sharing
***

Este repositorio trabaja con dos conjuntos de datos de **imágenes aéreas RGB de alta resolución** capturadas por dron en **Tapachula, Chiapas (México)**:

## 🗂️ SetDroneDataset_TapachulaRGB

Contiene información de **140 imágenes** etiquetadas manualmente a nivel de superpíxel en **10 categorías**: *árbol, suelo desnudo, pavimento, cuerpo de agua, techo de lámina, techo de losa, arbusto, hierba y pasto seco, hierba y pasto verde y sombra*.  
Este conjunto de datos se utilizó para **entrenar y validar** los modelos de aprendizaje automático.

🔗 Carpeta disponible en:  
https://drive.google.com/drive/folders/1VVvzgeX1ijy2it0-73B0oUnwz_FeZ896?usp=sharing

📍 **La carpeta contiene:**
- **140 imágenes** aéreas urbanas capturadas mediante dron en Tapachula, Chiapas (México).
- **Un archivo `.csv` por imagen** con estructura `Etiquetado_<nombre_imagen>.csv`, donde cada fila incluye:
  - el índice del superpíxel,
  - la categoría asignada,
  - y las **129 características** extraídas por superpíxel.
- **Un archivo `.tif` por imagen** que muestra la imagen segmentada y etiquetada.

---
## 🧹 Verificación y limpieza del etiquetado (superpíxeles por categoría)

Después de generar los conjuntos de datos. Se utilizo el script **verificacion_superpixeles_categoria.py**, para **inspeccionar la calidad del etiquetado manual** y detectar de forma rápida posibles errores o superpíxeles con mezcla de coberturas.

Este script genera, para cada imagen etiquetada, versiones en las que se resaltan únicamente los superpíxeles de **una categoría específica**, dejando el resto de la imagen con fondo negro. Las salidas se organizan automáticamente en **carpetas por categoría**, lo que facilita la revisión visual y la identificación de imágenes/categorías que requieren corrección manual.

**Ejemplo de salida (una imagen, múltiples categorías):**
- `Arbol/imagen_x5_1_arbol.jpg`
- `Techo_de_lamina/imagen_x5_1_techo_de_lamina.jpg`

Esta etapa de verificación fue utilizada como parte del preprocesamiento para asegurar un conjunto de datos más limpio antes del entrenamiento de los modelos.

---
    
EL archivo **instrucciones_ejecucion_notebooks.odt** contiene la información que requiere cada notebook para poder ser ejecutado correctamente. Donde cada ruta utilizada en los notebooks debe actualizarse según la ubicación local donde se encuentren almacenados los datos en cada equipo. 

Para ejecutar los notebooks, es necesario crear un **entorno de Python** con **Python 3.9.18** e instalar las librerías en las versiones indicadas para lograr compilar los notebooks de forma correcta. Estas dependencias se encuentran en el archivo **requerimientos_modelos.txt**.  
**Nota:** El entorno puede crearse con la herramienta de preferencia (por ejemplo, `venv`, `virtualenv` o Conda); lo importante es respetar la versión de Python requerida y las dependencias especificadas.

---

  
📌 **Licencia de uso:** Académico / Investigación no comercial.  



## 👩‍💻 Créditos

**Autores del código:**
- Karla Mauritania Reyes Maya  
- Viridiana Itzel Méndez Vásquez  
- Sarahi Ventura Angoa

**Supervisión académica:**
- Dr. Francisco Javier Hernández López (CIMAT Mérida)  
- Dr. Víctor Hugo Muñíz Sánchez (CIMAT Monterrey)

**Proyecto desarrollado en colaboración con:**
- Instituto Nacional de Salud Pública (INSP)  
- Dra. Kenya Mayela Valdez Delgado (contacto institucional)

---

© 2025 **Sarahi Ventura Angoa**  
Todos los derechos reservados. Este código no puede ser copiado, distribuido ni utilizado sin el permiso explícito de la autora.
