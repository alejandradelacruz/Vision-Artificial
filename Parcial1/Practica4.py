import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mahalanobis_distance(x, mean, cov):
    x_minus_mean = x - mean
    cov += np.eye(cov.shape[0]) * 1e-6  # para evitar problemas numéricos
    inv_cov_matrix = np.linalg.inv(cov)
    return np.sqrt(np.dot(np.dot(x_minus_mean, inv_cov_matrix), x_minus_mean))

def is_point_inside_cube(point, class_points):
    min_coords = np.min(class_points, axis=0)
    max_coords = np.max(class_points, axis=0)
    return np.all(point >= min_coords) and np.all(point <= max_coords)

def classify_point_max_prob(class_points, point_rgb):
    if not is_point_inside_cube(point_rgb, np.concatenate(class_points)):
        print("El punto de interés está fuera de los límites del cubo.")
        return

    max_likelihood = -np.inf
    closest_class_index = -1

    for i in range(N):
        class_rgb_points = np.array(class_points[i])
        class_mean = np.mean(class_rgb_points, axis=0)
        class_cov = np.cov(class_rgb_points, rowvar=False)

        # Manejar matrices singulares
        inv_covariance = np.linalg.inv(class_cov + np.eye(class_cov.shape[0]) * 1e-6)

        k = point_rgb.shape[0]
        delta = point_rgb - class_mean
        likelihood = (1 / ((2 * np.pi) ** (k / 2) * np.sqrt(np.linalg.det(class_cov)))) * \
                      np.exp(-0.5 * delta @ inv_covariance @ delta.T)

        if likelihood > max_likelihood:
            max_likelihood = likelihood
            closest_class_index = i

        print(f"Probabilidad de Clase {i + 1}: {likelihood}")
        print(f"Covarianza de Clase {i + 1}:\n{class_cov}\n")

    print(f"Clase más probable: Clase {closest_class_index + 1}")
    plot_points(class_points, point_rgb, closest_class_index)

def plot_points(class_points, point_rgb, closest_class_index):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N):
        class_rgb_points = np.array(class_points[i])
        ax.scatter(class_rgb_points[:, 0], class_rgb_points[:, 1], class_rgb_points[:, 2], label=f'Clase {i + 1}')
    ax.scatter(point_rgb[0], point_rgb[1], point_rgb[2], c='red', marker='x', label='Punto de Interés')
    
    ax.set_xlabel('Valor R')
    ax.set_ylabel('Valor G')
    ax.set_zlabel('Valor B')  
    ax.set_title('Puntos con Clases y Punto de Interés')
    ax.legend()
    plt.show()

N = 2  # Número de clases

# Clasificar puntos en el cubo
class1_points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
class2_points = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]

unknown_vector = np.array([float(input("Ingrese la coordenada x del vector desconocido: ")),
                           float(input("Ingrese la coordenada y del vector desconocido: ")),
                           float(input("Ingrese la coordenada z del vector desconocido: "))])

classify_point_max_prob([class1_points, class2_points], unknown_vector)
