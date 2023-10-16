import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_class_and_min_dist_euclidean(dataset_rgb, dataset_labels, classes_count, vector):
    min_distance = np.inf
    class_no = -1

    for class_index in range(1, classes_count + 1):
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index)
        mu = np.mean(class_values, axis=0)
        distance = np.linalg.norm(vector - mu)

        if distance < min_distance:
            min_distance = distance
            class_no = class_index

    return class_no, min_distance

def get_class_and_max_prob(dataset_rgb, dataset_labels, classes_count, vector):
    max_likelihood = -np.inf
    class_no = -1

    for class_index in range(1, classes_count + 1):
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index)
        mu = np.mean(class_values, axis=0)
        Sigma = np.cov(class_values, rowvar=False)
        k = len(vector)

        delta = vector - mu
        likelihood = (1 / ((2 * np.pi) ** (k / 2) * np.sqrt(np.linalg.det(Sigma)))) * \
                     np.exp(-0.5 * delta @ np.linalg.inv(Sigma) @ delta.T)

        if likelihood > max_likelihood:
            max_likelihood = likelihood
            class_no = class_index

    return class_no, max_likelihood

def get_class_and_min_dist_mahalannobis(dataset_rgb, dataset_labels, classes_count, vector):
    current_min = np.inf
    class_no = -1

    for class_index in range(1, classes_count + 1):
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index)
        mu = np.mean(class_values, axis=0)
        Sigma = np.cov(class_values, rowvar=False)
        Sigma_inv = np.linalg.inv(Sigma)

        delta = vector - mu
        D2 = delta @ Sigma_inv @ delta.T
        dist = np.abs(D2)

        if dist < current_min:
            current_min = dist
            class_no = class_index

    return class_no, current_min

def get_class_values(dataset_values, dataset_labels, desired_class):
    indices = np.where(dataset_labels == desired_class)[0]
    class_values = dataset_values[indices, :]
    return class_values


def point_is_in_image(x, y, img_size_x, img_size_y):
    return 1 <= x <= img_size_x and 1 <= y <= img_size_y

def get_n_points_inside_image_limits(c_grav_x, c_grav_y, img_size_x, img_size_y, elements_p_class):
    separated_factor = 30
    x_coordinates = np.random.randn(elements_p_class) * separated_factor + c_grav_x
    y_coordinates = np.random.randn(elements_p_class) * separated_factor + c_grav_y

    for i in range(elements_p_class):
        x_value = int(max(1, min(img_size_x, x_coordinates[i])))
        y_value = int(max(1, min(img_size_y, y_coordinates[i])))
        x_coordinates[i] = x_value
        y_coordinates[i] = y_value

    return x_coordinates.astype(int), y_coordinates.astype(int)

def get_rgb_from_coordinates(image, class_x_values, class_y_values, elements_p_class, class_no):
    dataset_rgb_values = np.zeros((elements_p_class, 3), dtype=np.uint8)
    dataset_labels = np.full(elements_p_class, class_no, dtype=np.int)

    for i in range(elements_p_class):
        rgb_value = image[class_y_values[i], class_x_values[i], :]
        dataset_rgb_values[i, :] = rgb_value

    return dataset_rgb_values, dataset_labels

def knn_euclidean(dataset_rgb, dataset_labels, k, vector):
    distances = np.linalg.norm(dataset_rgb - vector, axis=1)
    sorted_indices = np.argsort(distances)
    nearest_neighbors = sorted_indices[:k]
    neighbor_labels = dataset_labels[nearest_neighbors]
    class_no = np.bincount(neighbor_labels).argmax()
    return class_no

def get_conf_matrix_using_f(selected_criteria_function, no_classes, X, y, total_train_elements, k_for_knn):
    total_elements_count, _ = y.shape

    if total_train_elements == total_elements_count:
        train_data = X
        test_data = X
        train_labels = y
        test_labels = y
    else:
        train_data, train_labels, test_data, test_labels = get_test_train_data(X, y, total_train_elements)

    test_elements_count, _ = test_labels.shape
    conf_matrix = np.zeros((no_classes, no_classes), dtype=int)

    for element_no in range(test_elements_count):
        vector_x = test_data[element_no, :]
        expected_output = test_labels[element_no]
        predicted_class = -1

        if k_for_knn <= 0:
            predicted_class, _ = selected_criteria_function(train_data, train_labels, no_classes, vector_x)
        else:
            predicted_class = knn_euclidean(train_data, train_labels, k_for_knn, vector_x)

        conf_matrix[expected_output, predicted_class] += 1

    return conf_matrix

def leave_one_out_using_f(selected_criteria_function, no_classes, X, y, k_for_knn):
    total_elements_count, _ = y.shape
    conf_matrix = np.zeros((no_classes, no_classes), dtype=int)

    for element_no in range(total_elements_count):
        train_data = np.delete(X, element_no, axis=0)
        train_labels = np.delete(y, element_no, axis=0)

        test_data = X[element_no, :]
        test_labels = y[element_no]

        predicted_class = -1

        if k_for_knn <= 0:
            predicted_class, _ = selected_criteria_function(train_data, train_labels, no_classes, test_data)
        else:
            predicted_class = knn_euclidean(train_data, train_labels, k_for_knn, test_data)

        conf_matrix[test_labels, predicted_class] += 1

    return conf_matrix

def get_accuracy(conf_matrix):
    total_predictions = np.sum(conf_matrix)
    true_positives = np.trace(conf_matrix)
    accuracy = true_positives / total_predictions
    return accuracy

def get_test_train_data(dataset, labels, total_train_elements):
    classes = np.unique(labels)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    n_classes = len(classes)
    train_elements_per_class = total_train_elements // n_classes
    remainder = total_train_elements % n_classes

    for i in range(n_classes):
        class_indices = np.where(labels == classes[i])[0]
        class_data = dataset[class_indices, :]
        class_labels = labels[class_indices]

        idx = np.random.permutation(len(class_indices))
        class_data = class_data[idx, :]
        class_labels = class_labels[idx]

        if total_train_elements < n_classes:
            n_take = 1 if i < total_train_elements else 0
        else:
            n_take = train_elements_per_class + 1 if i < remainder else train_elements_per_class

        class_train_data = class_data[:n_take, :]
        class_train_labels = class_labels[:n_take]
        class_test_data = class_data[n_take:, :]
        class_test_labels = class_labels[n_take:]

        train_data.append(class_train_data)
        train_labels.append(class_train_labels)
        test_data.append(class_test_data)
        test_labels.append(class_test_labels)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)

    return train_data, train_labels, test_data, test_labels

def main():
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w', 'k']
    msj_no_classes = 'Ingresa el numero de clases a utilizar: '
    no_classes = int(input(msj_no_classes))
    msj_no_elementos_p_clase = 'Ingresa el representantes por clase: '
    no_elementos = int(input(msj_no_elementos_p_clase))

    no_elementos_dataset = no_elementos * no_classes
    classes_elements = np.zeros((no_elementos_dataset, 3), dtype=int)

    nombreImagen = 'C:\\Users\\User\\Desktop\\VA\\Vision-Artificial\\Parcial1\\Img\\peppers.png'
    playa = cv2.imread(nombreImagen)
    rows, cols, _ = playa.shape
    cv2.imshow('playa', playa)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    c_grav = np.zeros((no_classes, 2))
    dataset_rgb = np.zeros((no_elementos_dataset, 3), dtype=int)
    dataset_labels = np.zeros(no_elementos_dataset, dtype=int)
    counter = 0

    while counter < no_classes:
        cv2.imshow('Selecciona el centro de gravedad', playa)
        cv2.setMouseCallback('Selecciona el centro de gravedad', mouse_callback)
        cv2.waitKey(0)

        c_grav_x = c_grav[counter, 0]
        c_grav_y = c_grav[counter, 1]

        if not point_is_in_image(c_grav_x, c_grav_y, cols, rows):
            print('\n\nSelecciona solo puntos dentro de la imagen\n')
        else:
            cv2.circle(playa, (int(c_grav_x), int(c_grav_y)), 15, (0, 0, 0), -1)

            x_coordinates, y_coordinates = get_n_points_inside_image_limits(c_grav_x, c_grav_y, cols, rows, no_elementos)

            class_rgb_values, class_labels = get_rgb_from_coordinates(playa, x_coordinates, y_coordinates, no_elementos, counter + 1)

            start_idx = counter * no_elementos
            end_idx = start_idx + no_elementos
            dataset_rgb[start_idx:end_idx, :] = class_rgb_values
            dataset_labels[start_idx:end_idx] = class_labels

            color = colors[counter]
            for i in range(no_elementos):
                cv2.circle(playa, (x_coordinates[i], y_coordinates[i]), 5, color, -1)

            counter += 1

    user_input = 's'

    while user_input == 's':
        msg_selected_criteria = 'Ingresa el numero del criterio a utilizar para la clasificación: '
        print("Los criterios disponibles son los siguientes: \n1. Mahalannobis \n2. Distancia Euclidiana \n3. Máxima Probabilidad \n4. KNN")
        selected_criteria_idx = int(input(msg_selected_criteria))
        k_for_knn = -1
        selected_criteria_function = get_class_and_min_dist_mahalannobis
        selected_criteria_name = "Distancia Mahalanobis"

        if selected_criteria_idx == 2:
            selected_criteria_function = get_class_and_min_dist_euclidean
            selected_criteria_name = "Distancia Euclidiana"
        elif selected_criteria_idx == 3:
            selected_criteria_function = get_class_and_max_prob
            selected_criteria_name = "Criterio de Máxima probabilidad"
        elif selected_criteria_idx == 4:
            rgb_value_knn = np.array(input("Selecciona el vector en la imagen: "))
            k_for_knn = int(input("Teclea el número K para el cálculo del KNN (K > 0 e IMPAR): "))

            while k_for_knn % 2 == 0 or k_for_knn <= 0:
                k_for_knn = int(input("Teclea el número K para el cálculo del KNN (K > 0 e IMPAR): "))

            predicted_class = knn_euclidean(dataset_rgb, dataset_labels, k_for_knn, rgb_value_knn)
            print(f'Pertenece a la clase: {predicted_class}, color: {colors[predicted_class - 1]}')

        print(f"\nUsando {selected_criteria_name}")

        resustitution_conf_matrix = get_conf_matrix_using_f(selected_criteria_function, no_classes, dataset_rgb,
                                                            dataset_labels, no_elementos_dataset, k_for_knn)
        print("\nMATRIZ DE CONFUSIÓN")
        print(resustitution_conf_matrix)
        resustitution_accuracy = get_accuracy(resustitution_conf_matrix)

        iterations = 20
        print(f"CROSS-VALIDATION 50/50, {iterations} iteraciones")

        total_train_elements = no_elementos_dataset // 2
        cross_val_global_conf_matrix = np.zeros((no_classes, no_classes), dtype=int)

        for i in range(iterations):
            cross_val_conf_matrix = get_conf_matrix_using_f(selected_criteria_function, no_classes, dataset_rgb,
                                                            dataset_labels, total_train_elements, k_for_knn)
            print(f"\nMATRIZ DE CONFUSIÓN {i + 1}")
            print(cross_val_conf_matrix)
            cross_val_global_conf_matrix += cross_val_conf_matrix

        cross_val_accuracy = get_accuracy(cross_val_global_conf_matrix)

        print("\nLEAVE ONE OUT")
        leave_one_out_conf_matrix = leave_one_out_using_f(selected_criteria_function, no_classes, dataset_rgb,
                                                          dataset_labels, k_for_knn)
        print("\nMATRIZ DE CONFUSIÓN")
        print(leave_one_out_conf_matrix)
        leave_one_out_accuracy = get_accuracy(leave_one_out_conf_matrix)

        print(f"\nACCURACY USANDO {selected_criteria_name}")
        print(f"RESUSTITUCIÓN: {resustitution_accuracy}")
        print(f"CROSS-VALIDATION 50/50 ({iterations} ITERACIONES): {cross_val_accuracy}")
        print(f"LEAVE ONE OUT: {leave_one_out_accuracy}")

        x_bar = np.array([1, 2, 3])
        y_bar = np.array([resustitution_accuracy * 100 / no_elementos,
                          cross_val_accuracy * 100 / no_elementos,
                          leave_one_out_accuracy * 100 / no_elementos])

        cv2.imshow("Barras", np.zeros((100, 100, 3), dtype=np.uint8))
        plt.bar(x_bar, y_bar)
        plt.title("Barras")
        plt.show()

        user_input = input('¿Deseas usar otro criterio de clasificación? s: Continuar. Cualquier otra tecla: Salir')


if __name__ == "__main__":
    main()