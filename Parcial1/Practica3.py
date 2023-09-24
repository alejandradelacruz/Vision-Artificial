import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def perceptron(x, w):
    return np.dot(x, w)

X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

w = np.zeros(4)
for i in range(3):
    w[i] = float(input(f'Ingrese el valor del peso w{i + 1}: '))

w[3] = float(input('Ingrese el valor del peso w0: '))

# Factor de correlación
r = float(input('Ingrese el valor de r: '))

converge = False
etapa = 0

figura = plt.figure()
axis = figura.add_subplot(111, projection='3d')

while not converge:
    converge = True
    print(f'Étapa {etapa + 1}:')

    for i, x in enumerate(X):
        xn = np.append(x, 1)
        fsal = perceptron(xn, w)

        if fsal >= 0 and y[i] == 0:
            """ Formula para clase 1: XnTw >= 0 pertenece a la clase 1 """
            print(f'Para entrada {x}, w pertenece a la clase 1')
            """ Formula para corregir clase 1: wn+1 = wn - r * xn """
            w = w - r * xn
            print(f'Funcion de salida: ', fsal)
            converge = False
        elif fsal <= 0 and y[i] == 1:
            """ Formula para clase 2: XnTw <= 0 pertenece a la clase 2 """
            print(f'Para entrada {x}, w pertenece a la clase 2')
            """ Formula para corregir clase 2: wn+1 = wn + r * xn """
            w = w + r * xn
            print(f'Funcion de salida: ', fsal)
            converge = False
        else:
            print(f'Para entrada {x}, w se mantiene sin cambios')

    etapa = etapa + 1

print('\nEl perceptrón ha convergido:')
print(f'Valores finales de los pesos: {w}')

for i, x in enumerate(X):
    if y[i] == 0:
        axis.scatter(x[0], x[1], x[2], c='c', marker='o', label='Clase 1')
    else:
        axis.scatter(x[0], x[1], x[2], c='m', marker='x', label='Clase 2')

xx, yy = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
zz = (-w[0] * xx - w[1] * yy - w[3]) / w[2]
axis.plot_surface(xx, yy, zz, alpha=0.5)

axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')

axis.text(0, 0, 0, 'Clase 1', color='c')
axis.text(0, 0, 1, 'Clase 1', color='c')
axis.text(0, 1, 0, 'Clase 1', color='c')
axis.text(0, 1, 1, 'Clase 1', color='c')
axis.text(1, 0, 0, 'Clase 2', color='m')
axis.text(1, 0, 1, 'Clase 2', color='m')
axis.text(1, 1, 0, 'Clase 2', color='m')
axis.text(1, 1, 1, 'Clase 2', color='m')
plt.show()