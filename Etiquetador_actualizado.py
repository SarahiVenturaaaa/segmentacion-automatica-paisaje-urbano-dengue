import pathlib
import webbrowser
from tkinter import *
import tkinter as tk

# >> Links ttkbootstrap documentation
#
# https://ttkbootstrap.readthedocs.io/en/version-0.5/
# https://ttkbootstrap.readthedocs.io/en/version-0.5/widgets/button.html
import ttkbootstrap as ttk

from tkinter import filedialog
from tkinter import Tk, Button, Label
from PIL import Image
from PIL import ImageTk
import cv2
import imutils

import pandas as pd
import numpy as np
from scipy.stats import *
import os

import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import color


def recuperacion_etiquetado():
    '''
    En esta función se recupera una imagen ya etiquetada
    '''
    path_archivo_rec = filedialog.askopenfilename(filetypes = [
        ("archivo", ".csv")])
    etiquetado_data = pd.read_csv(path_archivo_rec)
    if etiquetado_data.empty:  # condicion por si damos clic en recuperar archivo y en realidad no hemos segmentado nadota
        print("El dataframe esta vacío")
        return
    cat_rec = etiquetado_data['categoria'] #Categoria de etiquetado
    label_rec = etiquetado_data['indice']
    tam = len(cat_rec)
    print('tamaño',tam)
    print(label_rec)

    global midataframe
    midataframe = etiquetado_data

    global Labels_slic
    labels = Labels_slic # matriz que contiene el numero del superpixel asignado

    # global image_seg
    # imge_ = image_seg  # copia de la imagen segmentada en superpixeles
    global respaldo_img
    imge_ = respaldo_img  # copia de la imagen segmentada en superpixeles

    erosion_size = 7
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))

    for i in range(0,tam):

        cat = cat_rec[i]
        indice = label_rec[i]
        print(indice)

        mask_p = labels == indice
        mask_p = mask_p.astype(dtype=np.uint8)
        mask_p = cv2.erode(mask_p, element_erode )
        Mask_p = mask_p*255

        respaldo = imge_
        ## Lo vamos a necesitar para ubicar los pixeles
        if(cat == 0):
            #'Arbol'
            #'Verde'
            r=69
            g=139
            b=0
        if(cat == 1):
            # 'Café'
            # 'Suelo Desnudo'
            #255,153,18
            r= 18
            g= 153
            b= 255
        if(cat == 2):
            # Gris
            # Pavimento
            # 104,131,139
            r = 139
            g = 131
            b = 104
        if(cat == 3):
            #Azul
            #Cuerpo de Agua
            # 61,89,171
            r = 171
            g = 85
            b = 61
        if(cat == 4):
            #Techo de Lamina
            #205,104,137)
            r=137
            g=104
            b=205
        if(cat == 5):
            #Techo loza
            #128,0,0
            r=0
            g=0
            b=128

        if(cat == 6):
            # Arbusto
            r = 85
            g = 107
            b = 47
        if(cat == 7):
            # Hierba y pasto seco
            r = 0
            g = 255
            b = 255
        if(cat == 8):
            # Hierba y pasto verde
            r = 124
            g = 252
            b = 0
        if(cat == 9):
            # Sombra
            r = 105
            g = 105
            b = 105
        if(cat == 10):
            # Sin etiqueta
            r = 0
            g = 0
            b = 0


        col,row = np.where(Mask_p[:,:]==255)
        respaldo[col,row,0] = r
        respaldo[col,row,1] = g
        respaldo[col,row,2] = b



        respaldo_img = respaldo


    imageToShowOutput_ = cv2.cvtColor(respaldo, cv2.COLOR_BGR2RGB)
    img_ = Image.fromarray(imageToShowOutput_)
    img_ = img_.resize((lblOutputImage.winfo_width(), lblOutputImage.winfo_height()))
    img_ = ImageTk.PhotoImage(image=img_)
    lblOutputImage.configure(image=img_)
    lblOutputImage.image = img_

    # >> Cambiar estilo del boton a success
    btn_a_r.config(style='success.TButton')

def elegir_archivo():

    path_archivo = filedialog.askopenfilename(filetypes = [
        ("archivo", ".txt")])


    if len(path_archivo) > 0:

        global archivo
        #archivo = open(path_archivo, 'r')
        #Lines = archivo.readlines()

        lista_line = []
        with open(path_archivo) as f:
            for line in f:
                line = line.partition('#')[0]
                line = line.rstrip()
                lista_line.append(line)
            print(lista_line)
        Lines = lista_line


        global parametro_1
        parametro_1 = Lines[0]

        global parametro_2
        parametro_2 = Lines[1]

        global parametro_3
        parametro_3 = Lines[2]

        global parametro_4
        parametro_4 = Lines[3]

        global categorias
        lista_c = Lines[4]
        lista_c = lista_c.split(",")
        categorias = lista_c
        print(lista_c)

        global nombre_arc
        name = Lines[5]
        nombre_arc = name
        print(name)


        selected = IntVar()

        global opciones
        opciones = lista_c


        # Limpia las opciones actuales
        opcion['menu'].delete(0, 'end')

        # Agregar las nuevas opciones
        for category in opciones:
            opcion['menu'].add_command(label=category, command=tk._setit(var, category))

    # Cambia la selección predeterminada
    # selected_option.set(nuevas_opciones[0])
    # >> Actualizar el estilo del boton
    btn_a.config(style='success.TButton')

def display_selection(*args):
    print(f"Opción seleccionada: {var.get()}")
    choice = var.get()
    # Label(root, text=choice).pack # >> Este label no debe aporta nada (es innecesario)
    print('etiqueta', choice)
    # var.get()
    if not choice in opciones: return
    inde = opciones.index(choice)
    print(inde)

    if inde == 0:
        # Arbol
        categoria = 0
    if inde == 1:
        # Suelo desnudo
        categoria = 1
    if inde == 2:
        # Pavimento
        categoria = 2
    if inde == 3:
        # Cuerpo de agua
        categoria = 3
    if inde == 4:
        # Techo de Lamina
        categoria = 4
    if inde == 5:
        # Techo de Loza
        categoria = 5
    if inde == 6:
        # Arbusto
        categoria = 6
    if inde == 7:
        # Hierba y pasto seco
        categoria = 7
    if inde == 8:
        # Hierba y pasto verde
        categoria = 8
    if inde == 9:
        # Sombra
        categoria = 9
    if inde == 10:
        # Sin etiqueta
        categoria = 10

    global categoria_et
    categoria_et = categoria
    # >> Llamar a la funcion para segmentar sobre la imagen
    etiqueta_segmento()

def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".tif"),
        ("image", ".png"),
        ("image", ".jpg")])

    global name_image
    name_image = os.path.splitext(os.path.basename(path_image))[0]
    print('name image', name_image)

    print('path',path_image)

    if len(path_image) > 0:
        # Insertar texto en el Entry
        file_path_text.config(state="normal")  # Cambiar temporalmente a modo editable para agregar texto
        file_path_text.insert(0, str(os.path.abspath(path_image)))
        file_path_text.config(state="readonly")

        global image
        image = cv2.imread(path_image)
        img_H = image.shape[0]
        img_W = image.shape[1]
        global tam_x
        tam_x = img_H
        global tam_y
        tam_y = img_W

        print('tamaño imagen',img_H)
        label_width = lblInputImage.winfo_width()
        label_height = lblInputImage.winfo_height()

        print(label_height, label_width)
        # Redimensionar la imagen al tamaño del Label
        imagen_redimensionada = cv2.resize(image, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        imageToShow = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow)
        img = ImageTk.PhotoImage(image=im)


        # Actualizar la imagen en el Label
        lblInputImage.config(image=img)
        lblInputImage.image = img


        # nueva_imagen = Image.open("ruta/a/nueva_imagen.png")  # Reemplaza con tu archivo
        # nueva_imagen = nueva_imagen.resize((200, 150), Image.ANTIALIAS)
        # nueva_photo = ImageTk.PhotoImage(nueva_imagen)
        # lblInputImage.config(image=nueva_photo)
        # lblInputImage.image = nueva_photo

        # Para visualizar la imagen de entrada en la GUI
        # imageToShow= imutils.resize(image, width=120)
        # imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        # im = Image.fromarray(imageToShow )
        # img = ImageTk.PhotoImage(image=im)
        #
        # lblInputImage.configure(image=img)
        # lblInputImage.image = img

        # Al momento que leemos la imagen de entrada, vaciamos
        # la iamgen de salida y se limpia la selección de los
        # radiobutton
        lblOutputImage.image = ""

        ###############################################################################
        ########################## FUNCIÓN DE SEGMENTACIÓN ############################
        ###############################################################################

        # Lo pasamos de BGR a grises
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Sobel Detección de Border
        #Horizontal: Dx
        sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_32FC1, dx=1, dy=0, ksize=5)
        #vertical:  Dy
        sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_32FC1, dx=0, dy=1, ksize=5)
        #Tomamos los valores absolutos
        Dx = abs(sobelx)
        Dy = abs(sobely)

        # Suma pesada con Dx y Dy
        MG = cv2.addWeighted(Dx, 0.5, Dy, 0.5, 0.0)
        Edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)

        global MG_
        MG_ = MG

        global Edges_
        Edges_ = Edges


        global parametro_1
        global parametro_2
        global parametro_3
        global parametro_4

        print(parametro_1)
        region_size_ = int(parametro_1)
        print(region_size_)
        ruler_ = int(parametro_2)
        print(ruler_)
        num_iterations = int(parametro_3)
        min_element_size = int(parametro_4)

        img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        slic = cv2.ximgproc.createSuperpixelSLIC (img_converted,region_size = region_size_, ruler =  ruler_)
        slic.iterate(num_iterations)

        if(min_element_size > 0):
            slic.enforceLabelConnectivity(min_element_size)
            #El tamaño mínimo del elemento en porcentajes que debe absorberse en un superpíxel más grand
            #### Obteniendo los contornos
        #### Devuelve la máscara de la segmentación de superpíxeles almacenada en el objeto SuperpixelSLIC
        #### La función devuelve los límites de la segmentación de superpíxeles.

        mask = slic.getLabelContourMask()
        dilation_size = 2

        #### Simplemente pasa la forma y el tamaño del kernel, obtiene el kernel deseado.
        #### Conde esta el ancla
        element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * dilation_size + 1, 2 * dilation_size + 1),(dilation_size, dilation_size))
        mask = cv2.dilate(mask,element_dilate)
        ## Esta dilatando

        label_slic = slic.getLabels()        # Obtener etiquetas de superpíxeles
        global Labels_slic
        Labels_slic = label_slic

        number_slic = slic.getNumberOfSuperpixels()  # Obtenga el número de superpíxeles
        mask_inv_slic = cv2.bitwise_not(mask)
        global img_slic
        img_slic = cv2.bitwise_and(image, image, mask =  mask_inv_slic)

        global image_seg
        image_seg = img_slic.copy()

        global respaldo_img
        respaldo_img = img_slic.copy()





        imageToShowOutput = cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(imageToShowOutput)
        img = img.resize((lblOutputImage.winfo_width(), lblOutputImage.winfo_height()))
        img = ImageTk.PhotoImage(image=img)
        lblOutputImage.configure(image=img)
        lblOutputImage.image = img


        # # Label IMAGEN DE SALIDA
        # lblInfo3 = Label(frame_right, text="Imagen Salida:", font="bold")
        # lblInfo3.grid(padx=5, pady=5)

        # >> Cambiar estilo del boton a success
        #
        #
        btn.config(style='success.TButton')
        btn_a_r.config(style='primary.TButton')
###############################################################################
########################### FUNCIÓN DE BORRAR #################################
###############################################################################

def borrado(event,SLIC):

    global respaldo_img
    respaldo = respaldo_img

    global image_seg
    imge_ = image_seg

    global midataframe
    datos = midataframe

    global Labels_slic
    labels = Labels_slic

    global tam_x
    Rate_x = tam_x/lblOutputImage.winfo_height()
    global tam_y
    Rate_y = tam_y/lblOutputImage.winfo_width()

    y_s = int(event.y)
    x_s = int(event.x)

    y_s = int(y_s*Rate_x)
    x_s = int(x_s*Rate_y)
    print('coordenada borrada',x_s,y_s)

    ### Aquí vamos a tener el indice para poder borrar y buscar donde lo estamos haciendo

    erosion_size =  7 # 7 era ek que tenia anteriormente 
    element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
    mask_p = labels == labels[y_s,x_s]
    mask_p = mask_p.astype(dtype=np.uint8)
    mask_p = cv2.erode(mask_p, element_erode )
    Mask_p = mask_p*255


    indice = labels[y_s,x_s]

    indices = list(midataframe.iloc[:, 1])
    print('indice en borrado',labels[y_s,x_s])
    if not indice in indices: return
    ind = indices.index(indice)
    print('borrado',ind)
    datos = datos.drop(ind,axis=0)
    datos.reset_index(inplace=True, drop=True)
    midataframe = datos

    print(datos)

    col,row = np.where(Mask_p[:,:] == 255)
    respaldo[col,row,:]= imge_[col,row,:]


    imageToShowOutput_ = cv2.cvtColor(respaldo, cv2.COLOR_BGR2RGB)
    img_ = Image.fromarray(imageToShowOutput_)
    img_ = img_.resize((lblOutputImage.winfo_width(), lblOutputImage.winfo_height()))
    img_ = ImageTk.PhotoImage(image=img_)
    lblOutputImage.configure(image=img_)
    lblOutputImage.image = img_

    # >> Guradar archivo .csv y .tif
    root.after(0, lambda: midataframe.to_csv(nombre_arc + '_' + name_image + '.csv', index=False))
    root.after(0, lambda: cv2.imwrite(nombre_arc + '_' + name_image + '.tif', respaldo))
###############################################################################
###############################################################################
######################### Función pinta pixeles ###############################
######################### Y recupera vector     ###############################

def coords(event,imagen,cat):

        global image
        global midataframe
        global Labels_slic
        labels = Labels_slic

        print (event.x,event.y)

        global tam_x
        Rate_x = tam_x/lblOutputImage.winfo_height()   # para ajustar el tamaño de la interfaz sin que ya este fijo 
        global tam_y
        Rate_y = tam_y/lblOutputImage.winfo_width()

        print("Label size:", lblOutputImage.winfo_width(), lblOutputImage.winfo_height())
        print("Image size:", tam_x, tam_y)

        y_s = int(event.y)
        x_s = int(event.x)
        # con esto conservamos el etiquetado sobre la imagen original (se multiplica porque anteriormente se dividia)
        y_s = int(y_s*Rate_x)
        x_s = int(x_s*Rate_y)

        print('coord en marc',x_s,y_s)

        print("Versión de OpenCV instalada:", cv2.__version__)

        ### Aquí vamos a tener el indice para poder borrar y buscar donde lo estamos haciendo
        ### Variable global
        print('Superpixel: {} | {}'.format(labels[y_s,x_s], cat))
        indice = labels[y_s,x_s]

        # >> Verificar si existe la categoria y el indice del superpixel
        #
        #
        state_category = ((midataframe['categoria'] == cat) & (midataframe['indice'] == indice)).any()
        if state_category: return
        print("State category: {}".format(state_category))
        erosion_size = 7
        element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))

        mask_p = labels == labels[y_s,x_s]
        mask_p = mask_p.astype(dtype=np.uint8)
        mask_p = cv2.erode(mask_p, element_erode )
        Mask_p = mask_p*255

        #cv2.imshow('Mascara',mat=Mask_p)
        row_e,col_e = np.where(Mask_p[:,:] == 255)

        global respaldo_img
        respaldo = respaldo_img

        mean_rgb = (round(np.mean(img_slic[row_e,col_e,2]),6),
                    round(np.mean(img_slic[row_e,col_e,1]),6),
                    round(np.mean(img_slic[row_e,col_e,0]),6))
        std_rgb = (round(np.std(img_slic[row_e,col_e,2]),6),
                   round(np.std(img_slic[row_e,col_e,1]),6),
                   round(np.std(img_slic[row_e,col_e,0]),6))


        gray_img = color.rgb2gray(image) # Se pasa la imagen a escala de grises
        gray_img = (gray_img*255).astype(dtype=np.uint8) #Se convierte en valores enteros de 0 a 255

        ############################################################
        # Para obtener el rectángulo orientado de menor área que
        # cubre al superpixel se utilizan las siguientes líneas
        ############################################################

        #Se obtiene el contorno de la máscara del superpixel
        contours, _ = cv2.findContours(Mask_p,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#find contour
        contour=contours[0]

        #Se obtiene el rectángulo de área mínima que cubre al superpixel
        rect = cv2.minAreaRect(contour) #Rectángulo
        box = cv2.boxPoints(rect) #Coordenadas de los vértices del rectángulo
        box = np.int0(box)

        # Con el fin de obtener el rectángulo orientado en la imagen, se recupera el ángulo de rotación
        centro, tamano, angulo = rect
        M = cv2.getRotationMatrix2D(centro, angulo, 1) #Matriz de rotación

        # Aplicar la matriz de transformación de rotación a la imagen para obtener la imagen y la máscara rotada
        mascara_rotada = cv2.warpAffine(Mask_p, M, (gray_img.shape[1], gray_img.shape[0])) #Máscara rotada
        imagen_rotada = cv2.warpAffine(gray_img, M, (gray_img.shape[1], gray_img.shape[0])) # Imagen en escala de grises rotada

        #Se obtiene el contorno de la máscara rotada
        contours_rot, _ = cv2.findContours(mascara_rotada,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_rot=contours_rot[0]

        #Rectángulo de área mínima de la imagen rotada
        rect_rot = cv2.minAreaRect(contour_rot)
        box_rot = cv2.boxPoints(rect_rot)
        box_rot = np.int0(box_rot)

        #Para cortar el rectángulo en la imagen rotada
        x, y, w, h = cv2.boundingRect(box_rot)

        # Recortar la región de interés
        gray_rect = imagen_rotada[y:y+h, x:x+w]


        ############################################################
        # Para obtener el rectángulo vertical/horizontal de menor área que
        # cubre al superpixel se utilizan las siguientes líneas
        ############################################################


        distances=[1,2,3,4,5]
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]

        global len_dist
        global len_ang

        len_dist = len(distances)
        len_ang = len(angles)



        # Se calcula la matriz de co-ocurrencia sobre el rectángulo que cubre al superpixel
        # Los ángulos son en radianes, si se ingresa una lista de ángulos se obtiene un tensor 4D
        # La tercer entrada es por distancia y la cuarta por ángulo
        glcm = graycomatrix(gray_rect, distances=distances, angles=angles, levels=256,
                        symmetric=True, normed=True)

        #Características obtenidas a partir de la matriz de co-ocurrencia
        dissimilarity = np.round(graycoprops(glcm, 'dissimilarity').reshape(-1, order='F'), 6)
        correlation = np.round(graycoprops(glcm, 'correlation').reshape(-1, order='F'), 6)
        contrast = np.round(graycoprops(glcm,'contrast').reshape(-1, order='F'), 6)
        homogeneity = np.round(graycoprops(glcm,'homogeneity').reshape(-1, order='F'),6)
        energy = np.round(graycoprops(glcm,'energy').reshape(-1, order='F'),6) #Es la raíz cuadrada de ASM: Angular Second Moment
        asm = np.round(graycoprops(glcm,'ASM').reshape(-1, order='F'),6)

        gray_feature = np.concatenate((dissimilarity,correlation,contrast,energy,homogeneity,asm))

        print(mean_rgb)
        ## Lo vamos a necesitar para ubicar los pixeles
        if(cat == 0):
            #'Arbol'
            #'Verde'
            r=69
            g=139
            b=0
        if(cat == 1):
            # 'naranja'
            # 'Suelo Desnudo'
            r= 18
            g= 153
            b= 255
        if(cat == 2):
            # Gris
            # Pavimento
            r = 139
            g = 131
            b = 104
        if(cat == 3):
            # Azul marino
            #Cuerpo de Agua
            r = 171
            g = 85
            b = 61
        if(cat == 4):
            #Techo de Lamina
            #rosa
            r=137
            g=104
            b=205
        if(cat == 5):
            #Techo loza
            # guinda
            r=0
            g=0
            b=128
        if(cat == 6):
            # Arbusto
            # oliva oscuro
            r = 85
            g = 107
            b = 47
        if(cat == 7):
            # Hierba y pasto seco
            # amarillo
            r = 0
            g = 255
            b = 255
        if(cat == 8):
            # Hierba y pasto verde
            # verde pistache
            r = 124
            g = 252
            b = 0
        if(cat == 9):
            # Sombra
            # gris
            r = 105
            g = 105
            b = 105
        if(cat == 10):
            # Sin etiqueta
            # negro
            r = 0
            g = 0
            b = 0


        col,row = np.where(Mask_p[:,:] == 255)
        ### El orden es BGR
        respaldo[col,row,0] = r
        respaldo[col,row,1] = g
        respaldo[col,row,2] = b

        #global respaldo_img
        #respaldo_img = respaldo

        global MG_
        MG = MG_
        MG_mean = np.mean(MG[row_e,col_e])
        MG_sdt = np.std(MG[row_e,col_e])  # corregido np.mean(np.std(MG[row_e,col_e])) 

        global Edges_
        Edges = Edges_

        locations_edges = cv2.findNonZero(Edges[row_e,col_e])
        locations_mask = cv2.findNonZero(mask_p)
        edge_density = round(np.size(locations_edges)/np.size(locations_mask),4)

        Vector_feature = {'categoria':cat,
                          'indice':indice,
                          'media_r':mean_rgb[0],
                          'media_g':mean_rgb[1],
                          'media_b':mean_rgb[2],
                          'std_r': std_rgb[0],
                          'std_g': std_rgb[1],
                          'std_b': std_rgb[2],
                          'mean_gb': MG_mean,
                          'std_mg':MG_sdt,
                          'density':edge_density
                          }

        features = list(Vector_feature.values())
        features.extend(list(gray_feature))


        ### Guardamos el vector de característias
        df_vector = pd.DataFrame(np.array(features).reshape(1, len(features)), columns=midataframe.columns)
        # Verificar si la fila existe
        if indice in midataframe['indice'].values:  # si ya existe el indice que se reemplace 
            midataframe.loc[midataframe['indice'] == indice] = df_vector.values
        else:
            midataframe = pd.concat([midataframe,df_vector],ignore_index=True)     


        global nombre_arc, name_image
        print(midataframe)

        # >> Agregar imagen segmentado al label
        #
        #
        imageToShowOutput_ = cv2.cvtColor(respaldo, cv2.COLOR_BGR2RGB)
        img_ = Image.fromarray(imageToShowOutput_)
        img_ = img_.resize((lblOutputImage.winfo_width(), lblOutputImage.winfo_height()))
        img_ = ImageTk.PhotoImage(image=img_)
        lblOutputImage.configure(image=img_)
        lblOutputImage.image = img_

        # >> Guradar archivo .csv y .tif, se guarda en paralelo 
        root.after(0, lambda: midataframe.to_csv(nombre_arc + '_' + name_image + '.csv', index=False))
        root.after(0, lambda: cv2.imwrite(nombre_arc+'_'+name_image+'.tif', respaldo))

###############################################################################
###############################################################################
######################### Función que segmenta ################################

def etiqueta_segmento():

    global categoria_et
    categoria = categoria_et

    global respaldo_img
    img_slic = respaldo_img


    global Labels_slic
    slic = Labels_slic

    imageToShowOutput = cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(imageToShowOutput)
    img = img.resize((lblOutputImage.winfo_width(), lblOutputImage.winfo_height()))
    img = ImageTk.PhotoImage(image=img)
    lblOutputImage.configure(image=img)
    lblOutputImage.image = img
    lblOutputImage.bind('<Button-1>',  lambda event, imagen = img_slic,cat=categoria: coords(event,imagen,cat))
    lblOutputImage.bind('<Button-3>',  lambda event, SLIC = slic : borrado(event,SLIC))
    # >> Conectar el evento de arrastre
    lblOutputImage.bind("<B1-Motion>", on_drag_segmentation)
    lblOutputImage.bind("<B3-Motion>", on_drag_delete)


    # Label IMAGEN DE SALIDA
    # lblInfo3 = Label(root, text="Imagen Salida:", font="bold")
    # lblInfo3.grid(column=1, row=1, padx=5, pady=5)

def nuevo_proceso():
    len_dist = 5
    len_ang = 4
    columns = ['categoria','indice','media_r','media_g','media_b','std_r','std_g','std_b','mean_gb','std_mg','density']

    features_gray = ['dissimilarity','correlation','contrast','energy','homogeneity','asm']

    columns_gray = [f'{feature}_dist{len}_ang{ang}' for feature in features_gray for len in range(len_dist) for ang in range(len_ang)]
    columns.extend(columns_gray)

    datos = pd.DataFrame(columns=columns)

    global midataframe
    midataframe = datos

    if img_slic.any():  # para evitar el error cuando no hemos etiquetaado ninguna imagen
        global respaldo_img
        respaldo_img = img_slic.copy()
        imageToShowOutput = cv2.cvtColor(respaldo_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(imageToShowOutput)
        img = img.resize((lblOutputImage.winfo_width(), lblOutputImage.winfo_height()))
        img = ImageTk.PhotoImage(image=img)
        lblOutputImage.configure(image=img)
        lblOutputImage.image = img

        # >> Guradar archivo .csv y .tif
        root.after(0, lambda: midataframe.to_csv(nombre_arc + '_' + name_image + '.csv', index=False))
        root.after(0, lambda: cv2.imwrite(nombre_arc + '_' + name_image + '.tif', respaldo_img))

###############################################################################
###############################################################################
#########################  EVENTOS DE TKINTER #################################
def on_resize(event):
    global resize_timer
    if resize_timer:
        lblOutputImage.after_cancel(resize_timer)
    resize_timer = lblOutputImage.after(1, lambda: update_image(event))

def update_image(event):
    # print("Event:", event.widget, "|", event.widget == lblOutputImage)
    if event.widget == lblOutputImage:
        lblOutputImage.config(text="")  # Eliminar texto
        lblOutputImage.config(image=None)
        # Mostrar el nuevo tamaño de la ventana
        # print(f"Nuevo tamaño: {event.width}x{event.height}")
        if respaldo_img.any():

            imageToShowOutput = cv2.cvtColor(respaldo_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(imageToShowOutput)
            img = img.resize((lblOutputImage.winfo_width(), lblOutputImage.winfo_height()))
            img = ImageTk.PhotoImage(image=img)

            lblOutputImage.configure(image=img)
            lblOutputImage.image = img
# Función para manejar el movimiento
# >> Segmentar
def on_drag_segmentation(event):
    print(f"Add segmentation: ({event.x_root}, {event.y_root})")
    coords(event, None, categoria_et)
# >> Eliminar segmentacion
def on_drag_delete(event):
    print(f"Del segmentation: ({event.x_root}, {event.y_root})")
    borrado(event, None)
    # coords(event, None, categoria_et)
def on_intput_img_dclick(event):
    file_path = file_path_text.get()
    print("File path: {}".format(file_path))
    if not os.path.isfile(file_path): 
        return
    webbrowser.open(pathlib.Path(file_path).as_uri())
###############################################################################
###############################################################################
#########################  VARIABLES GLOBALES #################################
img_slic = np.array([])
opciones = []
resize_timer = None
image = None
archivo  = None
respaldo_img = np.array([])
mat_prueba_ = None
image_seg = None
categoria_et = None
categorias = None
parametro_1 = None
parametro_2 = None
parametro_3 = None
parametro_4 = None
nombre_arc = None
name_image = None
tam_y = None
tam_x = None
MG_ = None
Edges_ = None
Labels_slic = None

len_dist = 5
len_ang = 4
columns = ['categoria','indice','media_r','media_g','media_b','std_r','std_g','std_b','mean_gb','std_mg','density']

columns_gray = []
features_gray = ['dissimilarity','correlation','contrast','energy','homogeneity','asm']

columns_gray = [f'{feature}_dist{len}_ang{ang}' for feature in features_gray for len in range(len_dist) for ang in range(len_ang)]
columns.extend(columns_gray)

midataframe = pd.DataFrame(columns=columns)


## Creamos la ventana
# >> Definir estilo # https://github.com/israel-dryer/ttkbootstrap/issues/57
# >> Estilos disponibles para window
# ['cosmo', 'flatly', 'litera', 'minty', 'lumen', 'sandstone', 'yeti', 'pulse', 'united', 'morph', 'journal', 'darkly', 'superhero', 'solar', 'cyborg', 'vapor', 'simplex', 'cerculean']
# root = ttk.Window(themename="sandstone") #
root = ttk.Window(themename="cosmo") #
# Cambiar el ícono de la ventana (archivo .ico)
#root.iconbitmap("icono.ico")  
# Configurar el grid
root.grid_columnconfigure(0, weight=0)  # Frame izquierdo no se expande
root.grid_columnconfigure(1, weight=1)  # Frame derecho se expande
root.grid_rowconfigure(0, weight=1)     # Expandir filas


style = ttk.Style()


root.title('Herramienta de Etiquedo.SLIC')
root.geometry('800x600')
# >> Definir tamaño mínimo a la interfaz gráfica
#
root.minsize(800, 600)
# >> Vincular el evento <Configure> a la ventana
# root.bind("<Configure>", on_resize)


# >> Agregar dos frames para seccionar los objetos
# >> Frame donde se mostrará la imagen
# frame_right = Frame(root)
# frame_right.pack(side="right", fill="both", expand=True)
frame_right = Frame(root, bg="lightgreen")
frame_right.grid(row=0, column=1, sticky="nsew")



# Frame superior
frame_up = tk.Frame(frame_right, bg="lightgray", height=80)
frame_up.pack(side="top", fill="x")
frame_up.grid_columnconfigure(0, weight=0)
frame_up.grid_columnconfigure(1, weight=1)

label_file_path = ttk.Label(frame_up, text="Ruta del archivo:", font=("TkDefaultFont", 8, "bold"))
label_file_path.grid(row=0, column=0, padx=0, pady=10)

file_path_text = ttk.Entry(frame_up, state="readonly", validate="focus", bootstyle="dark", font=("TkDefaultFont", 8, "bold"))
file_path_text.insert(0, "")
file_path_text.grid(row=0, column=1, padx=10, pady=10, sticky="ew")  # ew: Expandir de forma horizontal




# Frame inferior
frame_down = tk.Frame(frame_right, bg="black")
frame_down.pack(fill="both", expand=True)
# frame_down.grid(row=1, column=0, sticky="nsew")


lblOutputImage = Label(frame_down, borderwidth=2, relief="solid")
lblOutputImage.pack(expand=True, padx=10, pady=10, fill="both")
# >> Conectar funcion para actualizar el tamaño de la imagen sobre la etiqueta
lblOutputImage.bind("<Configure>", on_resize)
# °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°


# >> Frame donde se mostrarán las opciones para etiquetar
#
#
# Label ¿Qué categoría quieres etiquetar?
# frame_left = Frame(root, bg="lightblue", width=320, height=600)
# frame_left.pack(side="left", fill="y")
# frame_left.pack_propagate(False)
# frame_left.bind("<Configure>", limitar_tamano)
frame_left = Frame(root, bg="lightblue", width=320, height=600)
frame_left.grid(row=0, column=0, sticky="ns")  # Solo se expande verticalmente
frame_left.pack_propagate(False)

btn_a = ttk.Button(frame_left, text = '1.-Elegir archivo', width=25, command = elegir_archivo, style='secondary.TButton', takefocus=False)
btn_a.pack(padx=10, pady=10)

btn = ttk.Button(frame_left, text="2.-Elegir imagen", width=25, command = elegir_imagen, style='secondary.TButton', takefocus=False)
btn.pack(padx=10, pady=10)

# >> Agregar un separador horizontal
#
ttk.Separator(frame_left, orient='horizontal').pack(fill='x', padx=20, pady=10)

btn_a_r = ttk.Button(frame_left, text = '3.-Recuperar archivo', width=25, command = recuperacion_etiquetado, style='primary.TButton', takefocus=False)
btn_a_r.pack(padx=10, pady=10)

lblInfo2 = Label(frame_left, text="¿Qué categoría vas a marcar?", width=25)
lblInfo2.pack(padx=10, pady=10)

var = StringVar(root)
var.set('Categorías')
var.trace("w", display_selection)

opcion = OptionMenu(frame_left, var, "Categorías")
opcion.config(width=15)
opcion.pack(padx=10, pady=10)
# Label IMAGEN DE ENTRADA
lblInfo1 = Label(frame_left, text="Imagen Entrada:")
lblInfo1.pack(padx=10, pady=0)
# Label es donde cargamos la imagen de entrada
frame_aux = Frame(frame_left, width=280, height=150)
frame_aux.pack(padx=10, pady=0)
lblInputImage = Label(frame_aux, highlightthickness=2, highlightbackground="black")
lblInputImage.place(x=10, y=0, width=270, height=150)
# Asociar el evento de doble clic al Label
lblInputImage.bind("<Double-1>", on_intput_img_dclick)



lblInfo4 = Label(frame_left, text="Boton izquierdo: etiquetar")
lblInfo4.pack(pady=10)
lblInfo5 = Label(frame_left, text="Boton derecha: borrar")
lblInfo5.pack(pady=5)

# lblInputImage.place(x=10,y=270)
# °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

# >> Comandos para conectar funciones
# boton = Button(frame_left, text="Etiquetar", command = etiqueta_segmento )
# boton.pack(padx=10, pady=10)
# boton.place(x = 10, y = 200)


# >> Agregar un separador horizontal
#
ttk.Separator(frame_left, orient='horizontal').pack(fill='x', padx=10, pady=2)

boton_nuevo_pross = ttk.Button(frame_left, text="Nuevo Elemento", style='danger.TButton', command = nuevo_proceso, takefocus=False)
boton_nuevo_pross.pack(padx=10, pady=10)
# boton_nuevo_pross.place(x = 10, y = 500)

root.mainloop()
print("Version de OpenCV instalada:", cv2.__version__)

