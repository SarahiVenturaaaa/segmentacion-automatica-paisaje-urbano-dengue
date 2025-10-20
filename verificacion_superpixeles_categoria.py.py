import os
import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# Ocultar la ventana emergente de Tkinter
Tk().withdraw()

# Seleccionar las carpetas de entrada y salida
dir_carpeta_tif = filedialog.askdirectory(title="Selecciona la carpeta de entrada con los archivos .tif")
dir_carpeta_jpg = filedialog.askdirectory(title="Selecciona la carpeta de entrada con las imágenes originales .jpg")
dir_carpeta_salida = filedialog.askdirectory(title="Selecciona la carpeta de salida para guardar los superpixeles ")

# Colores asociados a cada categoría
colores_categorias = {
    0: (69, 139, 0),    # Árboles
    1: (18, 153, 255),  # Suelo desnudo
    2: (139, 131, 104), # Pavimento
    3: (171, 85, 61),   # Cuerpo de agua
    4: (137, 104, 205), # Techo de lámina
    5: (0, 0, 128),     # Techo de loza
    6: (85, 107, 47),   # Arbusto
    7: (0, 255, 255),   # Hierba y pasto seco
    8: (124, 252, 0),   # Hierba y pasto verde
    9: (105, 105, 105), # Sombra
    10: (128, 0, 128)   # Sin etiqueta
}

# Nombres de las carpetas asociadas a cada categoría
nombre_carpetas = {
    0: "Arboles",
    1: "Suelo Desnudo",
    2: "Pavimento",
    3: "Cuerpo de Agua",
    4: "Techo de Lamina",
    5: "Techo de Loza",
    6: "Arbusto",
    7: "Hierba y pasto seco",
    8: "Hierba y pasto verde",
    9: "Sombra",
    10: "Sin etiqueta"
}

# Procesar cada archivo .tif en la carpeta de entrada
for archivo in os.listdir(dir_carpeta_tif):
    if archivo.startswith('Etiquetado_') and archivo.endswith('.tif'):
        tif_path = os.path.join(dir_carpeta_tif, archivo)
        
        # Eliminar el prefijo 'Etiquetado_' para buscar la imagen original
        name_img_ori = archivo.replace('Etiquetado_', '').replace('.tif', '.jpg')
        original_image_path = os.path.join(dir_carpeta_jpg, name_img_ori)

        # Verificar que exista la imagen original correspondiente
        if not os.path.exists(original_image_path):
            print(f"No se encontró la imagen original para {archivo}. Saltando...")
            continue

        # Cargar la máscara y la imagen original
        mask_image = cv2.imread(tif_path, cv2.IMREAD_COLOR)
        original_image = cv2.imread(original_image_path)

        if mask_image is None or original_image is None:
            print(f"Error al cargar archivos para {archivo}. Saltando...")
            continue

        # Procesar cada categoría
        for categoria, color in colores_categorias.items():
            # Crear una máscara binaria para la categoría actual
            categoria_mask = cv2.inRange(mask_image, np.array(color), np.array(color))

            # Verificar si la categoría está presente en la máscara
            if np.sum(categoria_mask) == 0:
                print(f"Categoría {categoria} no encontrada en {archivo}.")
                continue

            # Aplicar la máscara directamente sobre la imagen original
            resultado_categoria = cv2.bitwise_and(original_image, original_image, mask=categoria_mask)

            # Crear carpeta para la categoría
            carpeta_categoria = os.path.join(dir_carpeta_salida, nombre_carpetas[categoria])
            os.makedirs(carpeta_categoria, exist_ok=True)

            # Guardar el resultado como una imagen
            name_superpixeles = f"{archivo.replace('Etiquetado_', '').replace('.tif', '')}_{nombre_carpetas[categoria]}.jpg"
            output_path = os.path.join(carpeta_categoria, name_superpixeles)
            cv2.imwrite(output_path, resultado_categoria)

print(f"Exportación completada. Los recortes se guardaron en: {dir_carpeta_salida}")
