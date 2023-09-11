import cv2
import numpy as np
import matplotlib.pyplot as plt

def mahalanobis_distance(x, mean, cov):
    x_minus_mean = x - mean
    cov += np.eye(cov.shape[0]) * 1e-6  # pa q no muera el calculo
    inv_cov_matrix = np.linalg.inv(cov)
    return np.sqrt(np.dot(np.dot(x_minus_mean, inv_cov_matrix), x_minus_mean))

def mouse_callback(event, x, y, flags, param):
    global clicked_points, current_class, current_click, class_means, class_covs, new_point_rgb, collecting_points, interest_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_class < N:
            rgb = image[y, x]
            print(f"Clic en RGB ({rgb[0]}, {rgb[1]}, {rgb[2]}) para la Clase {current_class + 1}")
            clicked_points[current_class].append((x, y, rgb))
            current_click += 1

            if current_click >= M:
                current_class += 1
                current_click = 0
        else:
            # Punto de interés
            rgb = image[y, x]
            new_point_rgb = (rgb[0], rgb[1], rgb[2])
            interest_points.append(new_point_rgb)
            # Verificar si el punto de interés está dentro del rango válido (0-255)
            if any(val < 0 or val > 255 for val in new_point_rgb):
                print("El punto de interés está fuera del rango de clases (0-255).")
            else:
                calculate_mahalanobis_distance(clicked_points, new_point_rgb)

def calculate_mahalanobis_distance(class_points, point_rgb):
    print(f"Calculando la distancia de Mahalanobis desde el punto de interés RGB {point_rgb} y encontrando la clase con la menor distancia...")

    distances = []
    for i in range(N):
        class_rgb_points = np.array([point[2] for point in class_points[i]])
        class_mean = np.mean(class_rgb_points, axis=0)
        class_cov = np.cov(class_rgb_points, rowvar=False)
        distance = mahalanobis_distance(point_rgb, class_mean, class_cov)
        distances.append(distance)
        print(f"Distancia de Mahalanobis a Clase {i + 1}: {distance}")

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    if min(distances) > mean_distance + 2 * std_distance:
        print("El punto de interés no pertenece a ninguna clase.")
    else:
        closest_class_index = np.argmin(distances)
        print(f"Clase más cercana: Clase {closest_class_index + 1}")
        plot_points(clicked_points, point_rgb, closest_class_index)

# Función para graficar los puntos y el punto de interés en un gráfico 3D
def plot_points(class_points, point_rgb, closest_class_index):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N):
        class_rgb_points = np.array([point[2] for point in class_points[i]])
        ax.scatter(class_rgb_points[:, 0], class_rgb_points[:, 1], class_rgb_points[:, 2], label=f'Clase {i + 1}')
    ax.scatter(point_rgb[0], point_rgb[1], point_rgb[2], c='red', marker='x', label='Punto de Interés')
    
    ax.set_xlabel('Valor R')
    ax.set_ylabel('Valor G')
    ax.set_zlabel('Valor B')  
    ax.set_title('Puntos con Clases y Punto de Interés')
    ax.legend()
    plt.show()

# Inicializa N, M y clicked_points fuera del bucle while
N = int(input("Ingrese el número de clases: "))  # Número de clases
M = int(input("Ingrese el número de representantes para las clases: "))  # Número de puntos de la imagen por clase
clicked_points = [[] for _ in range(N)]  # Una lista para cada clase
interest_points = []

def modify_interest_point(x, y):
    global interest_points
    if interest_points:
        rgb = image[y, x]
        new_rgb = (rgb[0], rgb[1], rgb[2])
        interest_points[0] = new_rgb
        print(f"Punto de interés modificado a RGB {new_rgb}")
        calculate_mahalanobis_distance(clicked_points,new_point_rgb)

image_path = 'AlfredoGucci.jpg'
image = cv2.imread(image_path)
while True: 
    current_class = 0
    current_click = 0
    new_point_rgb = None
    collecting_points = False 

    cv2.namedWindow('Imagen')
    cv2.setMouseCallback('Imagen', mouse_callback)
    cv2.imshow('Imagen', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Verifica si deseas modificar el punto de interés
    modify_point = input("¿Deseas modificar el punto de interés? (Sí/No): ").strip().lower()
    if modify_point == "si" or modify_point == "sí":    
        if interest_points:
            # Permite modificar el punto de interés haciendo clic en la imagen
            print("Haz clic en la imagen para modificar el punto de interés.")
            collecting_points = True
            cv2.namedWindow('Imagen')
            cv2.setMouseCallback('Imagen', mouse_callback)
            cv2.imshow('Imagen', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No hay un punto de interés para modificar.")
    elif modify_point == "no":
        # No se realiza ninguna modificación
        print("No se ha modificado el punto de interés por lo tanto nos vemos...")
        break

    repeat = input("¿Deseas reinicar el programa o salir? (Sí/No): ").strip().lower()
    
    if repeat != "si" and repeat != "sí":
        print("Ok Nos vemos...")
        break