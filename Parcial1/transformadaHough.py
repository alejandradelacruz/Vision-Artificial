# Importar paquetes útiles
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

imagen = mpimg.imread('Img/solidWhiteCurve.jpg')

# Imprimir algunas estadísticas y trazar la imagen
print('Esta imagen es:', type(imagen), 'con dimensiones:', imagen.shape)
plt.imshow(imagen)

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

# Aplicar filtro Gaussiano
tamaño_kernel = 9
gris_suavizado = cv2.GaussianBlur(gris, (tamaño_kernel, tamaño_kernel), 0)

# Configuración para Canny
umbral_bajo = 50
umbral_alto = 150
bordes = cv2.Canny(gris_suavizado, umbral_bajo, umbral_alto)

# Configuración para Hough
rho = 1
theta = np.pi/180
umbral_hough = 30
longitud_min_linea = 20
brecha_max_linea = 5
lineas = cv2.HoughLinesP(bordes, rho, theta, umbral_hough, np.array([]), longitud_min_linea, brecha_max_linea)

# Mostrar bordes detectados
plt.imshow(bordes)

# Crear una máscara utilizando cv2.fillPoly()
mascara = np.zeros_like(bordes)
color_mascara = 255

# Definir un polígono de cuatro lados para la máscara
forma_imagen = imagen.shape
vertices = np.array([[(0, forma_imagen[0]), (0, forma_imagen[0]*9/16), (forma_imagen[1], forma_imagen[0]*9/16), (forma_imagen[1], forma_imagen[0])]], dtype=np.int32)
cv2.fillPoly(mascara, vertices, color_mascara)
bordes = cv2.bitwise_and(bordes, mascara)

# Mostrar bordes después de aplicar la máscara
plt.imshow(bordes)

# Definir parámetros de transformada de Hough para líneas grandes


"""AJUSTAR PARAMETROS AQUI"""
# Crear una imagen en blanco del mismo tamaño que nuestra imagen para dibujar
rho = 1  # resolución de distancia en píxeles de la cuadrícula Hough
theta = np.pi/180  # resolución angular en radianes de la cuadrícula Hough
umbral_hough = 40  # número mínimo de votos (intersecciones en la celda de la cuadrícula Hough)
longitud_min_linea = 10  # número mínimo de píxeles que componen una línea
brecha_max_linea = 20  # brecha máxima en píxeles entre segmentos de línea conectables
imagen_lineas = np.copy(imagen)*0  # crear una imagen en blanco para dibujar líneas

# Aplicar transformada de Hough a la imagen de bordes
# El resultado "lineas" es una matriz que contiene los puntos finales de los segmentos de línea detectados
lineas = cv2.HoughLinesP(bordes, rho, theta, umbral_hough, np.array([]), longitud_min_linea, brecha_max_linea)

# Dibujar las líneas detectadas en la imagen original
imagen_mostrar = np.copy(imagen)

for linea in lineas:
    for x1, y1, x2, y2 in linea:
        cv2.line(imagen_mostrar, (x1, y1), (x2, y2), (255, 255, 0), 3)

plt.imshow(imagen_mostrar)
plt.show()
