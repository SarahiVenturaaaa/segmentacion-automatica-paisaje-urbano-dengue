#---------------------------------------------------------------------------
#    I M P O R T A C I O N     D E     L I B R E R I A S 

# ==== Interfaz gráfica ====
from tkinter import *  # Para crear interfaces gráficas con botones, ventanas, etc.
from PIL import Image  # Para abrir y manipular las imágenes
from PIL import ImageTk  # Para mostrar imágenes de PIL dentro de la interfaz de Tkinter

# ==== Procesamiento de imágenes ====
import cv2  # Librería OpenCV para procesamiento de imágenes (lectura, filtros, etc.)
import time  # Permite medir el tiempo o hacer pausas

# ==== Manejo y análisis de datos ====
import pandas as pd  # Para manejar datos en estructuras tipo tabla (DataFrames)
import numpy as np  # Para realizar operaciones numéricas y trabajar con arreglos

# ==== Estadística ====
from scipy.stats import *  # Para usar funciones estadísticas como distribuciones, pruebas, etc.

# ==== Sistema de archivos ====
import os  # Para interactuar con archivos y carpetas del sistema (leer rutas, verificar existencia, etc.)

# ==== Visualización ====
import matplotlib.pyplot as plt  # Para generar gráficos y visualizaciones
from matplotlib.patches import Patch  # Para crear leyendas personalizadas (cuadros de color, etc.)

# ==== Análisis de textura ====
from skimage.feature import graycomatrix, graycoprops  # Para calcular textura (GLCM y propiedades como contraste, energía, etc.)
from skimage import color  # Para conversiones de color, por ejemplo, a escala de grises

# ==== Morfología de imágenes ====
from skimage.morphology import disk  # Genera un elemento estructurante en forma de disco (útil en filtrado morfológico)
# from skimage.filters import median  # (Comentado) Aplica un filtro de mediana, útil para eliminar ruido

# ==== Preprocesamiento de datos ====
from sklearn.preprocessing import StandardScaler  # Estandariza datos (media = 0, desviación estándar = 1), útil para machine learning

# ==== Paralelización ====
from multiprocessing.pool import ThreadPool  # Permite ejecutar tareas en paralelo usando múltiples hilos (no procesos)
from multiprocessing import Pool, cpu_count # sé correctamente Pool.map con una lista de args

#---------------------------------------------
#  V A R I A B L E S   G L O B A L E S
# --------------------------------------------
global categorias, color_dict
# Diccionario de las categorias 
categorias = {
    0: "Árbol",
    1: "Suelo desnudo",
    2: "Pavimento",
    3: "Cuerpo de agua",
    4: "Techo de lámina",
    5: "Techo de losa",
    6: "Arbusto",
    7: "Hierba y pasto seco",
    8: "Hierba y pasto verde",
    9: "Sombra",
    10: "Sin etiqueta"
}


 # Diccionario de colores por categoría
color_dict = {
        'Árbol': (0,139,69), 
        'Suelo desnudo': (255, 153, 18), 
        'Pavimento': (104, 131, 139), 
        'Cuerpo de agua': (61, 85, 171), 
        'Techo de lámina': (205, 104, 137), 
        'Techo de losa': (128,0,0),
        'Arbusto': (47, 107, 85),
        'Hierba y pasto seco': (255, 255, 0),
        'Hierba y pasto verde': (0, 252, 124),
        'Sombra': (105, 105, 105),
        'Sin etiqueta': (0,0,0)
    }

def seg_SLIC(path_image, parametro_1= 40, parametro_2 = 10, parametro_3 = 10, parametro_4 = 20 ):
    '''
    Esta función realiza la segmentación de las imágenes mediante el método SLIC.
    Este código fue tomado del archivo Ultima_version.py proporcionado para el etiquetado.

    Entrada:
       path_image: ruta de la imagen a segmentar
       parametro 1 : Tamaño aproximado de los superpíxeles (más grande = menos superpíxeles)
       parametro 2 : Ruler: controla la compacidad (menor valor = más ajustado a bordes)
       parametro 3 : Número de iteraciones del algoritmo (mayor = más precisión)
       parametro 4 : Tamaño mínimo permitido para un superpíxel

    Salida:
        img: Imagen segmentada (formato PIL para mostrar)
        Labels_slic: matriz de etiquetas de cada superpíxel
        number_slic: número total de superpíxeles generados
    '''

    ######################################
    ## S E G M E N T A C I Ó N  (SLIC)
    ######################################

    # Parámetros para configurar la segmentación SLIC
    # parametro_1 = 40    # Tamaño aproximado de los superpíxeles (más grande = menos superpíxeles)
    # parametro_2 = 10    # Ruler: controla la compacidad (menor valor = más ajustado a bordes)
    # parametro_3 = 10    # Número de iteraciones del algoritmo (mayor = más precisión)
    # parametro_4 = 20    # Tamaño mínimo permitido para un superpíxel

    # Se guardan en variables locales con nombres descriptivos
    region_size_ = int(parametro_1)
    ruler_ = int(parametro_2)
    num_iterations = int(parametro_3)
    min_element_size = int(parametro_4)

    # Cargar imagen desde el path proporcionado
    image = cv2.imread(path_image)

    # Convertir la imagen de BGR a HSV para mejorar segmentación por color
    img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Crear el objeto SLIC para segmentar con los parámetros definidos
    slic = cv2.ximgproc.createSuperpixelSLIC(img_converted, region_size=region_size_, ruler=ruler_)

    # Ejecutar el algoritmo de segmentación SLIC
    slic.iterate(num_iterations)

    # Aplicar conectividad para eliminar regiones pequeñas no deseadas
    if min_element_size > 0:
        slic.enforceLabelConnectivity(min_element_size)
        # Se aseguran regiones mínimas conectadas, para evitar superpíxeles demasiado pequeños

    # Obtener la máscara de los bordes de los superpíxeles
    mask = slic.getLabelContourMask()

    # Definir tamaño de dilatación para engrosar los contornos
    dilation_size = 2
    element_dilate = cv2.getStructuringElement(
        cv2.MORPH_RECT, 
        (2 * dilation_size + 1, 2 * dilation_size + 1), 
        (dilation_size, dilation_size)
    )
    # Aplicar dilatación a los bordes para que se vean más claramente
    mask = cv2.dilate(mask, element_dilate)

    # Obtener etiquetas de cada superpíxel (matriz con misma forma que imagen)
    label_slic = slic.getLabels()
    Labels_slic = label_slic

    # Obtener el número de superpíxeles antes de ser absorvidos, cuando eran muy pequeños
    #number_slic = slic.getNumberOfSuperpixels()  # esto daba el numero
    
    # numero de superpixeles final
    number_slic= len(np.unique(Labels_slic))

    # Invertir la máscara (bordes en negro) para aplicar a la imagen original
    mask_inv_slic = cv2.bitwise_not(mask)

    # Aplicar la máscara invertida a la imagen original (se eliminan los contornos)
    img_slic = cv2.bitwise_and(image, image, mask=mask_inv_slic)

    # Convertir la imagen de BGR a RGB para mostrarla correctamente
    imageToShowOutput = cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB)

    # Convertir la imagen final a formato PIL (útil para mostrar en Tkinter)
    img_seg = Image.fromarray(imageToShowOutput)

    # Devolver imagen segmentada, etiquetas por superpíxel, y número total de superpíxeles
    return img_seg, Labels_slic, number_slic

def extrac_features_sec(path_image, rect='normal', std_data=False,
                        parametro_1=40, parametro_2=10, parametro_3=10, parametro_4=20):
#
    '''
    Función para segmentar una imagen y extraer características por superpíxel de forma secuencial
    
    Entradas:
        path_image: Ruta de la imagen a procesar.
        rect: Tipo de rectángulo para extracción de textura ('normal' o 'orientado').
        std_data: Booleano. Si es True, estandariza las características (media 0, varianza 1).
        parametro_1 (int): Tamaño aproximado de los superpíxeles.
        parametro_2 (float): Ruler, controla compacidad (menor = más ajustado).
        parametro_3 (int): Iteraciones para el algoritmo SLIC.
        parametro_4 (int): Tamaño mínimo permitido para un superpíxel

    Salidas:
        img_slic: Imagen con superpíxeles segmentados (para mostrar).
        Labels_slic: Matriz de etiquetas por superpíxel.
        number_slic: Número total de superpíxeles generados.
        all_features: DataFrame con características extraídas.
        total_time_extrac_carac_sec: Tiempo total de ejecución (seg).
        total_time_seg_slic: Tiempo de segmentación SLIC (seg).
        
    También guarda un archivo .csv con las características extraídas.
    '''
    
    # === INICIO DEl TIEMPO ===
    start_extrac_carac_sec = time.time()

    # === SEGMENTACIÓN CON SLIC ===
    start_seg_slic = time.time()
    img_slic, Labels_slic, number_slic = seg_SLIC(
        path_image= path_image,
        parametro_1=parametro_1,
        parametro_2=parametro_2,
        parametro_3=parametro_3,
        parametro_4=parametro_4
    )
    end_seg_slic = time.time()

    # === CARGA DE LA IMAGEN ORIGINAL ===
    image = cv2.imread(path_image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Imagen en escala de grises

    # === DETECCIÓN DE BORDES CON SOBEL ===
    sobelx = cv2.Sobel(img_gray, cv2.CV_32FC1, dx=1, dy=0, ksize=5)  # Derivada en X
    sobely = cv2.Sobel(img_gray, cv2.CV_32FC1, dx=0, dy=1, ksize=5)  # Derivada en Y
    Dx = abs(sobelx)
    Dy = abs(sobely)
    MG = cv2.addWeighted(Dx, 0.5, Dy, 0.5, 0.0)  # Magnitud del gradiente
    Edges = cv2.Canny(img_gray, 100, 200)  # Bordes con Canny

    # === CONFIGURACIÓN DE TEXTURA GLCM ===
    distances = [1, 2, 3, 4, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    len_dist = len(distances)
    len_ang = len(angles)

    # === COLUMNAS DEL DATAFRAME ===
    columns = ['indice','media_r','media_g','media_b','std_r','std_g','std_b','mean_gb','std_mg','density']
    features_gray = ['dissimilarity','correlation','contrast','energy','homogeneity','asm']
    columns_gray = [f'{feature}_dist{l}_ang{a}' for feature in features_gray for l in range(len_dist) for a in range(len_ang)]
    columns.extend(columns_gray)
    midataframe = pd.DataFrame(columns=columns)

    # === RECORRIDO POR CADA SUPERPÍXEL ===
    for ind in range(number_slic):

        # Crear máscara binaria del superpíxel
        erosion_size = 1
        element_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        mask_p = (Labels_slic == ind).astype(np.uint8)
        mask_p = cv2.erode(mask_p, element_erode)
        Mask_p = mask_p * 255

        # Extraer coordenadas de los píxeles activos
        row_e, col_e = np.where(Mask_p == 255)
        if len(row_e) == 0 or len(col_e) == 0:
            continue

        # === COLOR: Media y desviación estándar por canal ===
        mean_rgb = (
            round(np.mean(image[row_e, col_e, 2]), 6),
            round(np.mean(image[row_e, col_e, 1]), 6),
            round(np.mean(image[row_e, col_e, 0]), 6)
        )
        std_rgb = (
            round(np.std(image[row_e, col_e, 2]), 6),
            round(np.std(image[row_e, col_e, 1]), 6),
            round(np.std(image[row_e, col_e, 0]), 6)
        )

        # Imagen a escala de grises para textura
        gray_img = color.rgb2gray(image)
        gray_img = (gray_img * 255).astype(np.uint8)

        # === RECTÁNGULO PARA TEXTURA (normal u orientado) ===
        if rect == 'orientado':
            contours, _ = cv2.findContours(Mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            rect = cv2.minAreaRect(contour)
            centro, tamano, angulo = rect
            M = cv2.getRotationMatrix2D(centro, angulo, 1)
            mascara_rotada = cv2.warpAffine(Mask_p, M, gray_img.shape[::-1])
            imagen_rotada = cv2.warpAffine(gray_img, M, gray_img.shape[::-1])
            contours_rot, _ = cv2.findContours(mascara_rotada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            box_rot = cv2.boxPoints(cv2.minAreaRect(contours_rot[0]))
            x, y, w, h = cv2.boundingRect(np.int0(box_rot))
            gray_rect = imagen_rotada[y:y+h, x:x+w]
        else:
            gray_rect = gray_img[min(row_e):max(row_e), min(col_e):max(col_e)]

        if gray_rect.shape[0] == 0 or gray_rect.shape[1] == 0:
            continue

        # === TEXTURA: Matriz de co-ocurrencia y propiedades ===
        glcm = graycomatrix(gray_rect, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        dissimilarity = np.round(graycoprops(glcm, 'dissimilarity').reshape(-1, order='F'), 6)
        correlation = np.round(graycoprops(glcm, 'correlation').reshape(-1, order='F'), 6)
        contrast = np.round(graycoprops(glcm, 'contrast').reshape(-1, order='F'), 6)
        energy = np.round(graycoprops(glcm, 'energy').reshape(-1, order='F'), 6)
        homogeneity = np.round(graycoprops(glcm, 'homogeneity').reshape(-1, order='F'), 6)
        asm = np.round(graycoprops(glcm, 'ASM').reshape(-1, order='F'), 6)
        gray_features = np.concatenate((dissimilarity, correlation, contrast, energy, homogeneity, asm))

        # === DENSIDAD DE BORDES ===
        MG_mean = np.mean(MG[row_e, col_e])
        MG_sdt = np.std(MG[row_e, col_e])
        locations_edges = cv2.findNonZero(Edges[row_e, col_e])
        locations_mask = cv2.findNonZero(mask_p)
        edge_density = round(np.size(locations_edges) / np.size(locations_mask), 4)

        # === GUARDAR EN DATAFRAME ===
        dict_features = {
            'indice': ind,
            'media_r': mean_rgb[0],
            'media_g': mean_rgb[1],
            'media_b': mean_rgb[2],
            'std_r': std_rgb[0],
            'std_g': std_rgb[1],
            'std_b': std_rgb[2],
            'mean_gb': MG_mean,
            'std_mg': MG_sdt,
            'density': edge_density
        }

        features = list(dict_features.values()) + list(gray_features)
        df_vector = pd.DataFrame(np.array(features).reshape(1, len(features)), columns=midataframe.columns)
        midataframe = pd.concat([midataframe, df_vector], ignore_index=True)

    # === OPCIONAL: ESTANDARIZACIÓN DE CARACTERÍSTICAS ===
    features_ = midataframe.iloc[:, 1:]
    val_features = features_.values
    if std_data:
        ss = StandardScaler()
        sc = ss.fit_transform(val_features)
        all_features = pd.DataFrame(sc, columns=midataframe.columns[1:])
    else:
        all_features = pd.DataFrame(val_features, columns=midataframe.columns[1:])

    all_features['indice'] = midataframe['indice']

    # === GUARDAR ARCHIVO CSV DE CARACTERÍSTICAS ===
    name_image = os.path.splitext(os.path.basename(path_image))[0]
    all_features.to_csv('Caract_' + name_image + '.csv', index=False)

    # === TIEMPOS DE EJECUCIÓN ===
    end_extrac_carac_sec = time.time()
    total_time_extrac_carac_sec = np.round(end_extrac_carac_sec - start_extrac_carac_sec, 4)
    total_time_seg_slic = np.round(end_seg_slic - start_seg_slic, 4)

    return img_slic, Labels_slic, number_slic, all_features, total_time_extrac_carac_sec, total_time_seg_slic


# =====================================================================
#                   E X T R A E R   C A R A C T E R Í S T I C A S  
#                       E N   P A R A L E L O (PREPARACIÓN)
# =====================================================================

# Función que divide un rango total (por ejemplo, de superpíxeles)
# en n sub-rangos lo más equitativos posible, útil para paralelización.

def split_range(total, n):
    '''
    Esta función divide el total de superpíxeles en rangos, según el número de hilos,
    con el objetivo de repartir el trabajo de manera equitativa entre ellos.

    Entradas:
        total: número total  de superpíxeles)
        n: número de hilos (partes) en que se desea dividir el proceso.

    Salida:
        Lista de rangos [(inicio1, fin1), (inicio2, fin2), ..., (inicioN, finN)]
        donde cada rango es exclusivo al final (es decir, [inicio, fin)), que corresponde
        en cuanto se divide cada rango 
    '''

    # Cálculo del tamaño base para cada partición
    base = total // n         # división entera
    resto = total % n         # los elementos que sobran se distribuyen uno por uno

    # Genera una lista con los tamaños individuales de cada rango
    # A los primeros `resto` rangos se les suma 1 para repartir equitativamente
    tamanos = [base + 1 if i < resto else base for i in range(n)]

    # Generar los rangos de inicio y fin
    rangos = []
    inicio = 0
    for t in tamanos:
        fin = inicio + t
        rangos.append([inicio, fin])  # Rango: [inicio, fin) (fin no incluido)
        inicio = fin

    return rangos  # Por ejemplo: [[0, 4], [4, 8], [8, 11]] si en total hay 11 superpixeles y n=3

def parallel_slic(args):
    '''
    Función que se ejecuta en paralelo para extraer características de un rango de superpíxeles.

    Entrada:
        args: tupla/lista con los siguientes elementos:
            - slic: lista [inicio, fin) con el rango de superpíxeles a procesar.
            - Labels_slic: matriz de etiquetas por superpíxel (output de SLIC).
            - image: imagen original.
            - rect: 'normal' o 'orientado' para definir la forma del rectángulo GLCM.
            - distances: lista de distancias para GLCM.
            - angles: lista de ángulos para GLCM.
            - Edges: bordes detectados con Canny.
            - MG: magnitud del gradiente (Sobel).
            - columns: nombres de columnas del DataFrame.

    Salida:
        DataFrame con características extraídas para los superpíxeles procesados en ese hilo.
    '''
    # >>> Desempaquetar los 9 parámetros de entrada agrupados en 'args'
    # Esto permite que la función se ejecute con todos los elementos necesarios para la extracción
    slic, Labels_slic, image, rect, distances, angles, Edges, MG, columns = args

    # Lista para guardar los vectores de características generados por cada superpíxel
    rows = []

    # Desempaquetamos el rango de superpíxeles que le toca a este hilo
    initial, end = slic

    # Iteramos sobre cada índice de superpíxel asignado a este hilo
    for ind in range(initial, end):

        #  1: Crear una máscara binaria del superpíxel ===
        erosion_size = 1
        element_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (erosion_size, erosion_size))  # Kernel 3x3

        mask_p = (Labels_slic == ind).astype(np.uint8)  # Superpíxel binario (1s donde está presente)
        mask_p = cv2.erode(mask_p, element_erode)       # Se aplica erosión para "limpiar" bordes
        Mask_p = mask_p * 255                           # Se escala a rango 0–255

        # Se localizan las coordenadas de los píxeles pertenecientes al superpíxel
        row_e, col_e = np.where(Mask_p == 255)

        # Si el superpíxel no contiene píxeles (es muy pequeño o fue erosionado), lo ignoramos
        if len(row_e) == 0 or len(col_e) == 0:
            continue

        #  Calcula de la media y desviacion estandar para cada canal RGB 
        mean_rgb = (
            round(np.mean(image[row_e, col_e, 2]), 6),  # R
            round(np.mean(image[row_e, col_e, 1]), 6),  # G
            round(np.mean(image[row_e, col_e, 0]), 6)   # B
        )
        std_rgb = (
            round(np.std(image[row_e, col_e, 2]), 6),
            round(np.std(image[row_e, col_e, 1]), 6),
            round(np.std(image[row_e, col_e, 0]), 6)
        )

        #  Convertir la imagen a escala de grises
        gray_img = color.rgb2gray(image)
        gray_img = (gray_img * 255).astype(np.uint8)

        #  Obtener región rectangular del superpíxel
        if rect == 'orientado':
            # Rectángulo orientado (mínimo área rotado)

            contours, _ = cv2.findContours(Mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]

            rect = cv2.minAreaRect(contour)              # Devuelve centro, tamaño, ángulo
            box = np.int0(cv2.boxPoints(rect))           # Coordenadas de vértices

            centro, tamano, angulo = rect
            M = cv2.getRotationMatrix2D(centro, angulo, 1)

            mascara_rotada = cv2.warpAffine(Mask_p, M, (gray_img.shape[1], gray_img.shape[0]))
            imagen_rotada = cv2.warpAffine(gray_img, M, (gray_img.shape[1], gray_img.shape[0]))

            contours_rot, _ = cv2.findContours(mascara_rotada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_rot = contours_rot[0]

            rect_rot = cv2.minAreaRect(contour_rot)
            box_rot = np.int0(cv2.boxPoints(rect_rot))
            x, y, w, h = cv2.boundingRect(box_rot)

            gray_rect = imagen_rotada[y:y+h, x:x+w]      # Recorte de la imagen gris rotada
        else:
            #  Rectángulo horizontal/vertical (bounding box normal)
            gray_rect = gray_img[min(row_e):max(row_e), min(col_e):max(col_e)]

        # Si por alguna razón el recorte está vacío, continuamos con el siguiente superpíxel
        if gray_rect.shape[0] == 0 or gray_rect.shape[1] == 0:
            continue

        # Calcular la matriz de co-ocurrencia (GLCM)
        glcm = graycomatrix(
            gray_rect,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        #  Calcular las características de textura GLCM 
        dissimilarity = np.round(graycoprops(glcm, 'dissimilarity').reshape(-1, order='F'), 6)
        correlation = np.round(graycoprops(glcm, 'correlation').reshape(-1, order='F'), 6)
        contrast = np.round(graycoprops(glcm, 'contrast').reshape(-1, order='F'), 6)
        homogeneity = np.round(graycoprops(glcm, 'homogeneity').reshape(-1, order='F'), 6)
        energy = np.round(graycoprops(glcm, 'energy').reshape(-1, order='F'), 6)
        asm = np.round(graycoprops(glcm, 'ASM').reshape(-1, order='F'), 6)

        # Concatenamos todas las características texturales
        gray_features = np.concatenate((dissimilarity, correlation, contrast, energy, homogeneity, asm))

        # Calcular gradientes y densidad de bordes 
        MG_mean = np.mean(MG[row_e, col_e])              # Promedio de magnitud del gradiente
        MG_sdt = np.std(MG[row_e, col_e])                # Desviación estándar del gradiente
        locations_edges = cv2.findNonZero(Edges[row_e, col_e])   # Píxeles con borde detectado
        locations_mask = cv2.findNonZero(mask_p)                 # Todos los píxeles del superpíxel
        edge_density = round(np.size(locations_edges) / np.size(locations_mask), 4)

        #  diccionario con todas las características
        dict_features = {
            'indice': ind,
            'media_r': mean_rgb[0],
            'media_g': mean_rgb[1],
            'media_b': mean_rgb[2],
            'std_r': std_rgb[0],
            'std_g': std_rgb[1],
            'std_b': std_rgb[2],
            'mean_gb': MG_mean,
            'std_mg': MG_sdt,
            'density': edge_density
        }

        # Convertimos todo a un DataFrame con columnas estándar
        features = list(dict_features.values()) + list(gray_features)
        df_vector = pd.DataFrame(np.array(features).reshape(1, len(features)), columns=columns)

        # Agregamos el resultado de este superpíxel a la lista del hilo
        rows.append(df_vector)

    # Concatenar todas las filas procesadas en este hilo y devolver un DataFrame ===
    dataFrame = pd.concat(rows, ignore_index=True)
    return dataFrame


def extrac_parallel_features(path_image, rect='normal', std_data=False, n_threads = 3,
                             parametro_1=40, parametro_2= 10, parametro_3=10, parametro_4= 20):
    '''
    Función principal para segmentar una imagen y extraer características de superpíxeles en paralelo.
    
    Entradas:
        path_image: Ruta de la imagen a procesar.
        n_threads: Número de hilos a usar para paralelización (por defecto 3).
        rect: Tipo de recorte para GLCM ('normal' o 'orientado').
        std_data: Si es True, estandariza los vectores de características.
        parametro_1 (int): Tamaño aproximado de los superpíxeles.
        parametro_2 (float): Ruler, controla compacidad (menor = más ajustado).
        parametro_3 (int): Iteraciones para el algoritmo SLIC.
        parametro_4 (int): Tamaño mínimo permitido para un superpíxel


    Salidas:
        img_slic: Imagen segmentada en superpíxeles.
        Labels_slic: Etiquetas por superpíxel.
        number_slic: Número total de superpíxeles.
        all_features: DataFrame con las características extraídas.
        total_time_extrac_features_paralelo: Tiempo total de ejecución (segundos).
        total_time_seg_slic_paralelo: Tiempo que tomó la segmentación SLIC.
    '''

    # ====== MEDIR TIEMPO DE EJECUCIÓN TOTAL ======
    start_extrac_features_paralelo = time.time()

    # ====== SEGMENTACIÓN SLIC ======
    start_seg_slic_paralelo = time.time()
    img_slic, Labels_slic, number_slic = seg_SLIC(path_image=path_image,
        parametro_1=parametro_1,
        parametro_2=parametro_2,
        parametro_3=parametro_3,
        parametro_4=parametro_4)
    end_seg_slic_paralelo = time.time()

    # ====== CARGA DE IMAGEN ORIGINAL ======
    image = cv2.imread(path_image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ====== GRADIENTE DE BORDE CON SOBEL ======
    sobelx = cv2.Sobel(img_gray, cv2.CV_32FC1, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_32FC1, dx=0, dy=1, ksize=5)
    Dx = abs(sobelx)
    Dy = abs(sobely)
    MG = cv2.addWeighted(Dx, 0.5, Dy, 0.5, 0.0)  # Magnitud del gradiente
    Edges = cv2.Canny(img_gray, 100, 200)        # Bordes con Canny

    # ====== CONFIGURACIÓN GLCM ======
    distances = [1, 2, 3, 4, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    len_dist = len(distances)
    len_ang = len(angles)

    # ====== NOMBRES DE LAS COLUMNAS ======
    columns = ['indice', 'media_r', 'media_g', 'media_b',
               'std_r', 'std_g', 'std_b', 'mean_gb', 'std_mg', 'density']
    
    features_gray = ['dissimilarity', 'correlation', 'contrast',
                     'energy', 'homogeneity', 'asm']
    
    columns_gray = [f'{feature}_dist{l}_ang{a}'
                    for feature in features_gray
                    for l in range(len_dist)
                    for a in range(len_ang)]

    columns.extend(columns_gray)

    # ====== DIVIDIR EL TRABAJO ENTRE HILOS ======
    threads = split_range(number_slic, n_threads)
    # Ejemplo de salida: [[0, 100], [100, 200], [200, 300]] si hay 300 superpíxeles y 3 hilos

    # ====== EJECUTAR FUNCIONES EN PARALELO CON ThreadPool ======
    with ThreadPool(processes=n_threads) as pool:
        resultados = pool.map(
            parallel_slic,
            [(range_rows, Labels_slic, image, rect, distances, angles, Edges, MG, columns)
             for range_rows in threads]
        )

    # ====== CONCATENAR RESULTADOS DE TODOS LOS HILOS ======
    midataframe = pd.concat(resultados, ignore_index=True)

    # ====== OPCIONAL: ESTANDARIZACIÓN DE CARACTERÍSTICAS ======
    features_ = midataframe.iloc[:, 1:]  # Quitamos la columna 'indice'
    val_features = features_.values

    if std_data:
        ss = StandardScaler()
        sc = ss.fit_transform(val_features)
        all_features = pd.DataFrame(sc, columns=midataframe.columns[1:])
    else:
        all_features = pd.DataFrame(val_features, columns=midataframe.columns[1:])

    # Agregar de nuevo la columna 'indice'
    all_features['indice'] = midataframe['indice']

    # ====== GUARDAR RESULTADOS EN CSV ======
    name_image = os.path.splitext(os.path.basename(path_image))[0]
    all_features.to_csv('Caract_paralelo_' + name_image + '.csv', index=False)

    # ====== MEDIR TIEMPO FINAL ======
    end_extrac_features_paralelo = time.time()

    total_time_extrac_features_paralelo = np.round(end_extrac_features_paralelo - start_extrac_features_paralelo, 4)
    total_time_seg_slic_paralelo = np.round(end_seg_slic_paralelo - start_seg_slic_paralelo, 4)

    return img_slic, Labels_slic, number_slic, all_features, total_time_extrac_features_paralelo, total_time_seg_slic_paralelo


def pintar_bloque(rango, indices, prediction, Labels_slic, respaldo, categorias, color_dict, element_erode):
    '''
    Pinta los superpíxeles de un bloque de índices directamente sobre la imagen `respaldo`.
    '''
    for j in range(rango[0], rango[1]):
        mask_p = (Labels_slic == indices[j]).astype(np.uint8)
        mask_p = cv2.erode(mask_p, element_erode)
        Mask_p = mask_p * 255
        col, row = np.where(Mask_p == 255)

        nombre_categoria = categorias.get(prediction[j], "Sin etiqueta")
        b, g, r = color_dict[nombre_categoria]

        respaldo[col, row, 0] = b
        respaldo[col, row, 1] = g
        respaldo[col, row, 2] = r


def img_etiquetas_parallel(indices, prediction, Labels_slic, path_image, img_seg, num_threads=4):
    
    
    start_pintado_parallel= time.time()
    erosion_size = 1
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (erosion_size, erosion_size))

    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    respaldo = image.copy()

    # ==== Pintado paralelo por bloques con hilos ====
    rangos = split_range(len(indices), num_threads)
    args = [(rango, indices, prediction, Labels_slic, respaldo, categorias, color_dict, element_erode) for rango in rangos]

    with ThreadPool(num_threads) as pool:
        pool.starmap(pintar_bloque, args)

    image = Image.fromarray(respaldo)
    end_pintado_parallel= time.time()

    # ==== Calcular proporciones ====
    start_proporciones = time.time()
    proporciones_pixeles = calcular_proporcion_pixeles(indices, prediction, Labels_slic, image.size)
    end_proporciones = time.time()

    # ==== Leyenda ====
    legend_elements = [
        Patch(color=[r/255, g/255, b/255], label=f"{nombre}: {proporciones_pixeles.get(nombre, 0):.2f}%")
        for nombre, (r, g, b) in color_dict.items()
    ]

    # ==== Visualización ====
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(img_seg, alpha=0.3)
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Categorías y proporciones")
    ax.axis('off')
    plt.title('Imagen clasificada')

    name_image = os.path.splitext(os.path.basename(path_image))[0]
    plt.tight_layout()
    plt.savefig(name_image + '_clasificada.png', bbox_inches='tight', dpi=300)
    plt.show()

    total_time_pintado_parallel= np.round(end_pintado_parallel - start_pintado_parallel, 4)
    total_time_proporcion = np.round(end_proporciones - start_proporciones, 4)

    return total_time_pintado_parallel, total_time_proporcion



def img_etiquetas(indices, prediction, Labels_slic, path_image, img_seg):
    '''
    Asigna colores a los superpíxeles según las etiquetas de predicción y visualiza el resultado.

    Entradas:
        - indices: Lista con los índices de superpíxeles.
        - prediction: Lista de etiquetas predichas (números del 0 al 10):
            La asignación de colores esta basada en las predicciones del modelo con el que se esta trabajando

            Para cada superpíxel, el modelo genera una predicción numérica del 0 al 10, que se traduce en una categoría según el diccionario `categorias`.
            Luego, a cada categoría se le asigna un color en formato BGR utilizando el diccionario `color_dict`.

            Ejemplo de equivalencias:
                prediction[j] ==  0  →  'Árbol'               →  Verde oscuro
                prediction[j] ==  1  →  'Suelo desnudo'       →  Café
                prediction[j] ==  2  →  'Pavimento'           →  Gris
                prediction[j] ==  3  →  'Cuerpo de agua'      →  Azul
                prediction[j] ==  4  →  'Techo de lámina'     →  Rosa/morado
                prediction[j] ==  5  →  'Techo de losa'       →  Azul marino
                prediction[j] ==  6  →  'Arbusto'             →  Verde olivo
                prediction[j] ==  7  →  'Hierba y pasto seco' →  Amarillo
                prediction[j] ==  8  →  'Hierba y pasto verde'→  Verde claro
                prediction[j] ==  9  →  'Sombra'              →  Gris oscuro
                prediction[j] == 10  →  'Sin etiqueta'        →  Negro

            Esta lógica permite pintar visualmente cada superpíxel según la categoría predicha, facilitando la interpretación de los resultados.
        - Labels_slic: Matriz que indica a qué superpíxel pertenece cada píxel.
        - path_image: Ruta de la imagen original.
        - img_seg: Imagen segmentada (para superponerla de forma transparente).

    Salidas:
        total_time_pintado: Tiempo que tomó pintar los superpíxeles.
        total_time_proporcion: Tiempo que tomó calcular proporciones por categoría.
    '''

    # ========================
    # === CARGAR LA IMAGEN ===
    # ========================
    start_pintado_sec = time.time()
    erosion_size = 1
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (erosion_size, erosion_size))

    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    respaldo = image.copy()

    # ================================
    # === PINTAR CADA SUPERPÍXEL  ===
    # ================================
    for j in range(indices.shape[0]):
        mask_p = (Labels_slic == indices[j]).astype(np.uint8)
        mask_p = cv2.erode(mask_p, element_erode)
        Mask_p = mask_p * 255
        col, row = np.where(Mask_p == 255)

        # Aquí se asignan los colores según las predicciones del modelo.
        # Por ejemplo:
        #    Si prediction[j] == 0, entonces la categoría es 'Árbol' y se pinta de verde (color definido en el diccionario color_dict).
        #    Si prediction[j] == 1, corresponde a 'Suelo desnudo' y se pinta de color café (también desde color_dict).
        #    Y así sucesivamente con las demás categorías.
        # El color se recupera automáticamente a partir del número de predicción usando el diccionario 'categorias' para obtener el nombre,
        # y luego 'color_dict' para obtener el color correspondiente en formato BGR.

        nombre_categoria = categorias.get(prediction[j], "Sin etiqueta")
        b, g, r = color_dict[nombre_categoria]  # BGR
        
        ### El orden es BGR
        respaldo[col, row, 0] = b # Canal azul
        respaldo[col, row, 1] = g # Canal verde
        respaldo[col, row, 2] = r # Canal rojo

    image = Image.fromarray(respaldo)

    end_pintado_sec= time.time()

    # ========================================
    # === CALCULAR PROPORCIONES POR CLASE ===
    # ========================================
    start_proporciones = time.time()
    proporciones_pixeles = calcular_proporcion_pixeles(indices, prediction, Labels_slic, image.size)
    end_proporciones = time.time()

    # ========================================
    # === CREAR LEYENDA CON PORCENTAJES    ===
    # ========================================
    legend_elements = [
        Patch(color=[r/255, g/255, b/255], label=f"{nombre}: {proporciones_pixeles.get(nombre, 0):.2f}%")
        for nombre, (r, g, b) in color_dict.items()
    ]

    # ========================================
    # === VISUALIZAR Y GUARDAR RESULTADO  ===
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(img_seg, alpha=0.3)
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Categorías y proporciones")
    ax.axis('off')
    plt.title('Imagen clasificada')

    # Guardar la imagen
    plt.tight_layout()
    name_image = os.path.splitext(os.path.basename(path_image))[0]
    plt.savefig(name_image + '_clasificada.png', bbox_inches='tight', dpi=300)
    plt.show()

    # ===============================
    # === TIEMPOS DE EJECUCIÓN    ===
    # ===============================
    total_time_pintado_sec= np.round(end_pintado_sec - start_pintado_sec, 4)
    total_time_proporcion = np.round(end_proporciones - start_proporciones, 4)

    return total_time_pintado_sec, total_time_proporcion






def calcular_proporcion_pixeles(indices, prediction, Labels_slic, img_shape):
    """
    Calcula la proporción (%) de píxeles por categoría en una imagen clasificada.

    Parámetros:
        indices: Arreglo de índices de superpíxeles a procesar (por ejemplo, [0, 1, 2, ..., N]).
        prediction: Lista de categorías predichas (índices de 0 a 10).
        Labels_slic: Matriz con etiquetas por píxel (cada valor indica a qué superpíxel pertenece).
        img_shape: Dimensiones de la imagen (alto, ancho), aunque aquí no se usa directamente.

    Salida:
        Diccionario con categorías como llaves y proporciones en porcentaje como valores.
    """


    # === Inicializar conteo de píxeles por categoría
    pixeles_por_categoria = {cat: 0 for cat in categorias.values()}

    # === Recorrer cada superpíxel para acumular su conteo de píxeles
    for j in range(indices.shape[0]):
        mask_p = Labels_slic == indices[j]               # Máscara binaria del superpíxel j
        num_pixeles = np.sum(mask_p)                     # Total de píxeles activos (True = 1)

        categoria_nombre = categorias.get(prediction[j], "Sin etiqueta")
        pixeles_por_categoria[categoria_nombre] += num_pixeles

    # === Sumar todos los píxeles clasificados
    total_pixeles = sum(pixeles_por_categoria.values())

    # === Calcular proporciones por categoría (porcentaje)
    if total_pixeles > 0:
        proporcion_pixeles = {
            cat: (pixeles_por_categoria[cat] / total_pixeles) * 100
            for cat in pixeles_por_categoria
        }
    else:
        # Si no hay píxeles clasificados (evitar división por cero)
        proporcion_pixeles = {cat: 0 for cat in pixeles_por_categoria}

    return proporcion_pixeles


def ver_superpixeles_categoria(indices, predicciones, Labels_slic, path_image, categoria_objetivo=10):
    '''
    Muestra los superpíxeles de una categoría específica en color original,
    y pinta el resto en negro. También muestra cuántos superpíxeles corresponden.

    Parámetros:
        indices: Índices de superpíxeles.
        predicciones: Etiquetas predichas por superpíxel, obtenidas por el modelo previamente entrenado 
        Labels_slic: Matriz de etiquetas por píxel (SLIC).
        path_image: Ruta de la imagen original.
        categoria_objetivo: Índice de la categoría a visualizar.
    '''


    # Convertir categoría objetivo a nombre
    nombre_categoria = categorias.get(categoria_objetivo, "Name_categoria")

    # Cargar imagen original
    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Imagen negra
    salida = np.zeros_like(image)

    # Contador de superpíxeles que cumplen la condición
    contador = 0
    for j in range(indices.shape[0]):
        if predicciones[j] == categoria_objetivo:
            mask = (Labels_slic == indices[j])
            salida[mask] = image[mask]
            contador += 1

    # Mostrar resultado
    plt.imshow(salida)
    plt.title(f'Hay {contador} superpíxeles etiquetados como: {nombre_categoria}')
    plt.axis('off')
    plt.show()



