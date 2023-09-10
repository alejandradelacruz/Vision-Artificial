import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Define los datos de entrada d1, m1, m2, y la matriz inversa de d1
d1 = np.array([[-3/4, 1/4, 1/4, 1/4], [-1/4, -1/4, -1/4, 3/4], [-1/4, -1/4, 3/4, -1/4]])
m1 = np.array([[3/4, 1/4, 1/4]])
m2 = np.array([[1/4, 3/4, 3/4]])
d1_t = np.transpose(d1)
matriz_inversa = np.linalg.inv(1/4 * np.dot(d1, d1_t))

# Solicita al usuario que ingrese los valores para xbusca
xbusca_x = float(input("Ingrese el valor de x menor a 1 para x: "))
xbusca_y = float(input("Ingrese el valor de y menor a 1 para y: "))
xbusca_z = float(input("Ingrese el valor de z menor a 1 para z: "))

# Verifica si xbusca está dentro del cubo [0, 1] en todas las dimensiones
if 0 <= xbusca_x <= 1 and 0 <= xbusca_y <= 1 and 0 <= xbusca_z <= 1:
    # Crea el vector xbusca con los valores ingresados por el usuario
    xbusca = np.array([[xbusca_x, xbusca_y, xbusca_z]])

    # Calcula matrizfinald1 y matrizfinald2 para xbusca
    restad1 = (xbusca - m1)
    restad2 = (xbusca - m2)
    restatransd1 = np.transpose(restad1)
    restatransd2 = np.transpose(restad2)
    auxd1 = np.dot(restad1, matriz_inversa)
    auxd2 = np.dot(restad2, matriz_inversa)
    matrizfinald1 = np.sqrt(np.dot(auxd1, restatransd1))
    matrizfinald2 = np.sqrt(np.dot(auxd2, restatransd2))

    print("Distancia1")
    print(1/4 * np.dot(d1, d1_t))
    print(matriz_inversa)
    print(matrizfinald1)
    print("Distancia2")
    print(1/4 * np.dot(d1, d1_t))
    print(matriz_inversa)
    print(matrizfinald2)
    # Comparación de las matrices y asignación a una clase
    if np.trace(matrizfinald1) > np.trace(matrizfinald2):
        clase_asignada = 2
    elif np.trace(matrizfinald1) < np.trace(matrizfinald2):
        clase_asignada = 1
    else:
        clase_asignada=0

    # Imprime la clase asignada
    if clase_asignada==0:
        print(f"El punto no pertenece a ninguna clase, por lo tanto se le asigna 0")
    else:
        print(f"El punto xbusca se asigna a la Clase {clase_asignada}")
else:
    print("El punto xbusca está fuera del cubo y no pertenece a ninguna clase")

# Define los vértices del cubo, incluyendo el nuevo vértice [1/2, 1/2, 1/2]
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                     [xbusca_x, xbusca_y, xbusca_z]])  # Nuevo vértice

# Etiqueta de clase para cada vértice (asumamos que hay 9 vértices y 3 clases)
clases = [1, 1, 1, 2, 2, 1, 2, 2, clase_asignada if 0 <= xbusca_x <= 1 and 0 <= xbusca_y <= 1 and 0 <= xbusca_z <= 1 else 0]

# Crea la figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotea las aristas del cubo excluyendo las aristas de la base
aristas = [[vertices[0], vertices[1]], [vertices[1], vertices[2]],
           [vertices[2], vertices[3]], [vertices[3], vertices[0]],
           [vertices[4], vertices[5]], [vertices[5], vertices[6]],
           [vertices[6], vertices[7]], [vertices[7], vertices[4]],
           [vertices[0], vertices[4]], [vertices[1], vertices[5]],
           [vertices[2], vertices[6]], [vertices[3], vertices[7]]]

for arista in aristas:
    ax.add_collection3d(Line3DCollection([arista], colors='black', linewidths=1))

# Plotea los vértices como puntos etiquetados con sus clases
for vertice, clase in zip(vertices, clases):
    ax.scatter(vertice[0], vertice[1], vertice[2], c='red', marker='o', s=50)
    ax.text(vertice[0], vertice[1], vertice[2], f'Clase {clase}', fontsize=8, color='black', ha='center', va='bottom')

# Define los límites del gráfico
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Etiqueta los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Muestra el gráfico
plt.show()