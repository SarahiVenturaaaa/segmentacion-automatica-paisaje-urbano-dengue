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

Se desarrolló una herramienta de segmentación semántica del paisaje urbano en Tapachula, Chiapas, utilizando técnicas de aprendizaje automático. Su propósito es identificar de forma automática distintas coberturas del paisaje, como vegetación, cuerpos de agua, techos, pavimento, entre otras, que están asociadas a factores de riesgo para la transmisión de enfermedades como el dengue.

Para ello, se entrenaron modelos de clasificación automática que permiten categorizar superpíxeles en imágenes aéreas urbanas segmentadas, facilitando la detección de condiciones ambientales relacionadas con la presencia del mosquito *Aedes aegypti*


## 📂 Contenido del repositorio

- Código para:
  - Segmentación de imágenes.
  - Etiquetado manual y automático de superpíxeles.
  - Extracción de características estadísticas, de bordes, de textura y multiespectrales.
  - Clasificación automática de superpíxeles en categorías de interés.
- Entrenamiento y evaluación de modelos:
  - K-Nearest Neighbors (KNN)
  - Regresión Logística (RL)
  - Máquinas de Soporte Vectorial (SVM)
  - Random Forest (RF)
  - Redes Neuronales Multicapa (MLP)
- Asignación automática de la categoría **“Sin etiqueta”** para superpíxeles no clasificados.
- Visualización de resultados y predicciones sobre las imágenes segmentadas.
- Reporte técnico detallado y notebooks explicativos.

## 🛠 Requisitos

- Python 3.9 o superior  
- Bibliotecas necesarias:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `joblib`

Se incluyen archivos `.txt` con las versiones exactas de las librerías requeridas para facilitar la reproducción del entorno.


🗂️ Dataset utilizado: TapachulaRGB-DroneDataset

Este repositorio trabaja con un conjunto de datos propio denominado:

📌 TapachulaRGB: Conjunto de datos de imágenes aéreas RGB obtenidas por dron en Tapachula, Chiapas

🔗 Acceso público al dataset:
https://drive.google.com/drive/folders/1VVvzgeX1ijy2it0-73B0oUnwz_FeZ896?usp=sharing

📍 Características principales:

Imágenes aéreas urbanas capturadas mediante dron en Tapachula, Chiapas (México)

Resolución alta apta para segmentación por superpíxeles

Contiene escenas con:

Vegetación (árboles, arbustos, pasto seco y verde)

Techos (losa y lámina)

Pavimento y suelo desnudo

Zonas con sombra

Cuerpos de agua

Se incluyen versiones:

Originales RGB

Segmentadas con sus máscaras

Archivos .csv con etiquetas y características por superpíxel

📌 Licencia de uso: Académico / Investigación no comercial
📌 Si utilizas estos datos en un trabajo académico, por favor cita:

Ventura, S. (2025). TapachulaRGB: Conjunto de datos de imágenes aéreas RGB obtenidas por dron en Tapachula, Chiapas [Dataset].
Disponible en: https://drive.google.com/drive/folders/1VVvzgeX1ijy2it0-73B0oUnwz_FeZ896?usp=sharing

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
