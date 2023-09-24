import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def perceptron_activation(x, w):
    return np.dot(x, w)

X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 1, 1, 1, 1, 1])

w = np.zeros(4)  
for i in range(4):
    w[i] = float(input(f'Ingrese el valor del peso w{i + 1}: '))

r = float(input('Ingrese el valor del factor de correlación (r): '))

converge = False
etapa = 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while not converge:
    converge = True
    print(f'Étapa {etapa + 1}:')
    
    for i, x in enumerate(X):
        x_extended = np.append(x, 1)  #
        activation = perceptron_activation(x_extended, w)
        
        if activation >= 0 and y[i] == 0:
            print(f'Para entrada {x}, w pertenece a la clase 1')
            w = w - r * x_extended
            converge = False
        elif activation <= 0 and y[i] == 1:
            print(f'Para entrada {x}, w pertenece a la clase 2')
            w = w + r * x_extended
            converge = False
        else:
            print(f'Para entrada {x}, w se mantiene sin cambios')
    
    etapa = etapa + 1

print('\nEl perceptrón ha convergido:')
print(f'Valores finales de los pesos: {w}')

# Graficar el espacio de clases final en 3D
for i, x in enumerate(X):
    if y[i] == 0:
        ax.scatter(x[0], x[1], x[2], c='r', marker='o', label='Clase 1')
    else:
        ax.scatter(x[0], x[1], x[2], c='b', marker='x', label='Clase 2')

xx, yy = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
zz = (-w[0] * xx - w[1] * yy - w[3]) / w[2]
ax.plot_surface(xx, yy, zz, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()