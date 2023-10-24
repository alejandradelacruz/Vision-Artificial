import numpy as np
import matplotlib.pyplot as plt

def perceptron(x, w):
    return np.dot(x, w)

# Nuevos puntos para Clase 1
X_clase1 = np.array([[1, 0], [2, 0]])

# Nuevos puntos para Clase 2
X_clase2 = np.array([[-2, 0], [-1, 2]])

X = np.concatenate((X_clase1, X_clase2))
y = np.array([0, 0, 1, 1])

w = np.zeros(3)
for i in range(2):
    w[i] = float(input(f'Ingrese el valor del peso w{i + 1}: '))

w[2] = float(input(f'Ingresa el valor de w0: '))

r = float(input('Ingrese el valor del coeficiente de error r: '))
if r <= 0:
    print('El coeficiente de error debe ser mayor a 0.')
    exit()

converge = False
etapa = 0

plt.figure()

while not converge:
    converge = True
    print(f'Étapa {etapa + 1}:')

    for i, x in enumerate(X):
        xn = np.append(x, 1)
        fsal = perceptron(xn, w)

        if fsal >= 0 and y[i] == 0:
            print(f'Para entrada {x}, pertenece a la clase 1')
            w = w - r * xn  # Corrección para clase 1
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
        plt.scatter(X[i][0], X[i][1], c='red', marker='o', label='Clase 1')
    else:
        plt.scatter(X[i][0], X[i][1], c='blue', marker='x', label='Clase 2')

plt.xlabel('X')
plt.ylabel('Y')

xx = np.linspace(-3, 3, 2)
yy = (-w[0] * xx - w[2]) / w[1]
plt.plot(xx, yy, label='Línea de decisión')

plt.legend()
plt.show()