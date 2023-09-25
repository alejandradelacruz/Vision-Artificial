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

r = float(input('Ingrese el valor del coeficiente de error r: '))
if r <= 0:
    print('El coeficiente de error debe ser mayor a 0.')
    exit()

converge = False
etapa = 0

fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')

while not converge:
    converge = True
    print(f'Étapa {etapa + 1}:')

    for i, x in enumerate(X):
        xn = np.append(x, 1)
        fsal = perceptron(xn, w)

        if fsal >= 0 and y[i] == 0:
            print(f'Para entrada {x}, pertenece a la clase 1')
            w = w - r * xn  #Correcion para clase 1
            converge = False
        elif fsal <= 0 and y[i] == 1: 
            print(f'Para entrada {x}, pertenece a la clase 2')
            w = w + r * xn  # Corrección para clase 2
            converge = False
        else:
            print(f'Para entrada {x}, permanece sin cambios')
    etapa = etapa + 1

print('\nEl perceptrón ha convergido:')
print(f'Valores finales de los pesos: {w}')

for i in range(len(X)):
    if y[i] == 0:
        axis.scatter(X[i][0], X[i][1], X[i][2], c='red', marker='o', label='Clase 1')
    else:
        axis.scatter(X[i][0], X[i][1], X[i][2], c='blue', marker='x', label='Clase 2')

axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')

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