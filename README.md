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


Este repositorio contiene el conjunto de datos, los notebooks, herramientas y documentación generados durante la consultoría realizada en colaboración con el Instituto Nacional de Salud Pública (INSP).

Se desarrolló una herramienta de segmentación semántica del paisaje urbano en Tapachula, Chiapas, utilizando un enfonque basado en **superpíxeles (SLIC)** y **modelos de aprendizaje automático**, con el objetivo de identificar de forma automática distintas coberturas del paisaje, como vegetación, cuerpos de agua, techos, pavimento, entre otras, que están asociadas a factores de riesgo para la transmisión de enfermedades como el dengue (mosquito *Aedes aegypti*).

## 📂 Contenido del repositorio VERSION NUEVA

- **Segmentación** de imágenes en superpíxeles (SLIC).
- **Etiquetado manual** de superpíxeles (herramienta gráfica) y generación de archivos `.csv`/`.tif` por imagen.
- **Extracción de características** por superpíxel:
  - Estadísticas de color (RGB),
  - medidas de bordes/estructura (p. ej., densidad),
  - texturas GLCM (dissimilarity, correlation, contrast, energy, homogeneity, ASM) en múltiples distancias y ángulos.
- **Entrenamiento y evaluación** de modelos de clasificación:
  - K-Nearest Neighbors (KNN),
  - Regresión Logística (RL),
  - Support Vector Machines (SVM),
  - Random Forest (RF),
  - Multilayer Perceptron (MLP).
- **Estrategias de incertidumbre** para asignar la categoría **“Sin etiqueta”** a superpíxeles ambiguos/no representados (basadas en probabilidades; se documentan las estrategias evaluadas y la seleccionada).
- **Clasificación automática completa de imágenes** (flujo de punta a punta):
  - segmentación + extracción paralela + predicción por superpíxel con un modelo entrenado,
  - cálculo de **probabilidades por clase** y aplicación de la estrategia de **“Sin etiqueta”**,
  - generación de salidas por imagen (`Caract_paralelo_[imagen].csv`) con predicción original y predicción final,
  - visualizaciones: superpíxeles por categoría y gráficas de proporciones por clase (barras y pastel),
  - evaluación y reportes: matriz de confusión y métricas globales/por clase (cuando aplica).

## 📂 Contenido del repositorio VERSION VIEJA:

- Código para:
  - Segmentación de imágenes.
  - Etiquetado manual y automático de superpíxeles.
  - Extracción de características estadísticas, de bordes, y de textura.
  - Clasificación automática de superpíxeles en categorías de interés.
- Entrenamiento y evaluación de modelos:
  - K-Nearest Neighbors (KNN)
  - Regresión Logística (RL)
  - Máquinas de Soporte Vectorial (SVM)
  - Random Forest (RF)
  - Redes Neuronales Multicapa (MLP)
- Asignación automática de la categoría **“Sin etiqueta”** para superpíxeles no clasificados.
- Visualización de resultados y predicciones sobre las imágenes segmentadas.
- Clasificacion automatica de las imagenes para cada uno de los modelos de aprendizaje etnteenad

## 🛠 Requisitos

- Python 3.9 o superior  
- Bibliotecas necesarias:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `joblib`

El archivo **requerimentos_etiquetador.txt** contiene la versión de las librerías requeridas para ejecutar el script **Etiquetador_actualizado.py**.

Este repositorio trabaja con dos conjuntos de datos de **imágenes aéreas RGB de alta resolución** capturadas por dron en **Tapachula, Chiapas (México)**.

---

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

## 🗂️ test_set_rgb

Contiene información de **23 imágenes** etiquetadas manualmente a nivel de superpíxel en **11 categorías**: las 10 anteriores más **“Sin etiqueta”**.  
Este conjunto se utilizó para **evaluar el desempeño** de los modelos entrenados de forma más realista, así como para la **construcción y evaluación de estrategias** para asignar la categoría **“Sin etiqueta”**, ya que los modelos fueron entrenados originalmente sin conocimiento de esta clase adicional.

🔗 Carpeta disponible en:  
https://drive.google.com/drive/folders/1K6Irsw0WHXWKRMfe4bm82g1zk5-7LHjs?usp=sharing

📍 **La carpeta contiene:**
- **23 imágenes** originales utilizadas como conjunto de prueba.
- **Un archivo `.csv` por imagen** con estructura `Etiquetado_<nombre_imagen>.csv`, donde cada fila incluye:
  - el índice del superpíxel,
  - la categoría asignada,
  - y las **129 características** extraídas por superpíxel.
- **Un archivo `.tif` por imagen** que muestra la imagen segmentada y etiquetada.

---
    
EL archivo **instrucciones_ejecucion_notebooks.odt** contiene la información que requiere cada notebook para poder ser ejecutado correctamente. Donde cada ruta utilizada en los notebooks debe actualizarse según la ubicación local donde se encuentren almacenados los datos en cada equipo.

---

  
📌 **Licencia de uso:** Académico / Investigación no comercial.  
📌 **Citación recomendada:**

> Ventura, S. (2025). *TapachulaRGB: Conjunto de datos de imágenes aéreas RGB obtenidas por dron en Tapachula, Chiapas* [Dataset].  
> Disponible en: https://drive.google.com/drive/folders/1VVvzgeX1ijy2it0-73B0oUnwz_FeZ896?usp=sharing


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
