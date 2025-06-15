# segmentacion automatica paisaje urbano dengue
Herramienta para segmentación semántica del paisaje urbano en Tapachula, Chiapas, mediante aprendizaje automático, para identificar coberturas del paisaje de forma automática  (vegetación, cuerpos de agua, techos, pavimento, etc.) asociados con el riesgo de transmisión de enfermedades como el dengue


Segmentación semántica del paisaje en zonas urbanas para la determinación de factores de riesgo de transmisión de dengue
Este repositorio contiene los notebooks, herramientas y documentación desarrollados durante la consultoría con el Instituto Nacional de Salud Pública (INSP).

Se entrenaron modelos de aprendizaje automático para clasificar superpíxeles en imágenes aéreas urbanas segmentadas, con el objetivo de identificar condiciones del paisaje asociadas a la presencia del mosquito Aedes aegypti.

Contenido del repositorio
Código para:

Segmentación de imágenes.

Etiquetado manual y automático de superpíxeles.

Extracción de características multiespectrales, estadísticas, texturales y de bordes.

Clasificación automática de superpíxeles en categorías de interés.

Entrenamiento y evaluación de modelos:

K-Nearest Neighbors (KNN)

Regresión Logística (RL)

Máquinas de Soporte Vectorial (SVM)

Random Forest (RF)

Redes Neuronales Multicapa (MLP)

Asignación automática de categoría “Sin etiqueta” para superpíxeles no clasificados.

Visualización de resultados y predicciones sobre las imágenes segmentadas.

Reporte técnico detallado y notebooks explicativos.

Requisitos
Python 3.9 o superior

Bibliotecas necesarias:

scikit-learn

numpy

pandas

matplotlib

joblib

Se incluyen archivos .txt con las versiones exactas de las librerías requeridas para facilitar la reproducción del entorno.

Créditos
Autores del código:

Karla Mauritania Reyes Maya

Viridiana Itzel Méndez Vásquez

Sarahi Ventura Angoa

Supervisión académica:

Dr. Francisco Javier Hernández López (CIMAT Mérida)

Dr. Víctor Hugo Muñíz Sánchez (CIMAT Monterrey)

Proyecto desarrollado en colaboración con:

Instituto Nacional de Salud Pública (INSP)

Dra. Kenya Mayela Valdez Delgado (contacto institucional)

© 2025 Sarahi Ventura Angoa
Todos los derechos reservados. Este código no puede ser copiado, distribuido ni utilizado sin el permiso explícito de la autora.
