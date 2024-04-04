
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.spatial import distance
import matplotlib.image as mpimg

def loadImage(numero):
    __path__ = 'C:\\Users\\User\\Desktop\\VA\\Vision-Artificial\\Parcial3\\img'
    img_path = f"{__path__}\\{numero}.jpg"

    if cv2.imread(img_path) is not None: 
        img = cv2.imread(img_path)
        return img
    else:
        print(f"La imagen {numero}.jpg no existe en la carpeta especificada.")
        return None

def binarize(img_array, umbral=128):
    _, img_binaria = cv2.threshold(img_array, umbral, 255, cv2.THRESH_BINARY)
    return img_binaria

def cleanImage(img_binaria):
    # Morphological opening Erod->Dilate
    kernel_e = np.ones((2, 2), np.uint8)
    kernel_di = np.ones((5, 5), np.uint8)
    erode_img = cv2.erode(img_binaria, kernel_e, iterations=1)
    clean_img = cv2.dilate(erode_img, kernel_di, iterations=1)
    return clean_img

def countObjects(matriz_binaria):
    # Etiqueta las regiones conectadas
    labeled_array, num_features = label(matriz_binaria)
    return labeled_array, num_features

def calculateAPC(clean_image):
    contours, _ = cv2.findContours(
        clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_areas = []
    object_perimeters = []
    object_centroids = []

    for contour in contours:
        area = cv2.contourArea(contour)
        object_areas.append(round(area, 2))

        perimeter = cv2.arcLength(contour, closed=True)
        object_perimeters.append(round(perimeter, 2))

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            object_centroids.append((cX, cY))

    return object_areas, object_perimeters, object_centroids

def extract_features(contours, centroids):
    features = []
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        centroid = centroids[i]
        features.append([area, perimeter, centroid[0], centroid[1]])
    return features

def train_max_probability_classifier(features, labels):
    max_probability_classifier = MultinomialNB()
    max_probability_classifier.fit(features, labels)
    return max_probability_classifier

def train_euclidean_classifier(features, labels):
    def euclidean_classifier(x, y):
        # Asigna la etiqueta del objeto según su posición en la lista de características
        return labels[features.index(x)]

    return euclidean_classifier, features

def mahalanobis_classifier(features, labels):
    class_means = {}
    class_covariance = {}

    # Calcular la media y covarianza de cada clase
    for label in set(labels):
        class_features = [features[i]
                          for i in range(len(features)) if labels[i] == label]
        class_means[label] = np.mean(class_features, axis=0)

        # Calcular la matriz de covarianza con regularización
        cov = np.cov(class_features, rowvar=False)
        reg_cov = cov + np.identity(cov.shape[0]) * 1e-6
        class_covariance[label] = reg_cov

    def mahalanobis_distance(x, mean, cov):
        x_minus_mean = x - mean
        cov_inv = np.linalg.inv(cov)
        return np.sqrt(np.dot(np.dot(x_minus_mean, cov_inv), x_minus_mean))

    def predict(sample):
        distances = {}
        for label, mean in class_means.items():
            distances[label] = mahalanobis_distance(
                sample, mean, class_covariance[label])
        return min(distances, key=distances.get)

    return predict

def train_knn_classifier(features, labels, min_neighbors=1, max_neighbors=10):
    num_objects = len(set(labels))
    n_neighbors = min(max(min_neighbors, num_objects), max_neighbors)

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(features, labels)
    return knn_classifier

def showImages(original, cleaned_image, labeled_array, numero_imagen, obj_areas, obj_perimeters, obj_centroids, object_labels, object_names):
    labeled_array = labeled_array.astype(np.uint8)

    imagen_coloreada = np.zeros(
        (cleaned_image.shape[0], cleaned_image.shape[1], 3), dtype=np.uint8)
    colores = np.zeros((labeled_array.max() + 1, 3), dtype=np.uint8)
    colores[:, 1] = 255  # Establecer el canal verde al máximo

    print("\nEntrenar imagenes con Max Prob.\n")

    for label_number in range(1, labeled_array.max() + 1):
        labeled_region = (labeled_array == label_number)
        imagen_coloreada[labeled_region] = colores[label_number]

        area, perim, centroid = obj_areas[label_number -
                                          1], obj_perimeters[label_number - 1], obj_centroids[label_number - 1]
        object_label = object_labels[label_number - 1]
        object_name = object_names.get(
            object_label, 'Tipo de objeto desconocido')
        x, y = 0, 0
        text_position = (x, (y + (label_number * 15)))
        # cv2.putText(imagen_coloreada, f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        print(
            f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}')
        cv2.putText(imagen_coloreada, f'{label_number}', centroid, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)


    # Muestra la imagen
    plt.imshow(imagen_coloreada)
    plt.imshow(imagen_coloreada)
    plt.show()

def showImagesEuclidean(original, cleaned_image, labeled_array, numero_imagen, obj_areas, obj_perimeters, obj_centroids, object_labels, object_names):
    labeled_array = labeled_array.astype(np.uint8)

    imagen_coloreada = np.zeros(
        (cleaned_image.shape[0], cleaned_image.shape[1], 3), dtype=np.uint8)
    colores = np.random.randint(
        50, 226, (labeled_array.max() + 1, 3), dtype=np.uint8)

    print("\nEntrenar imagenes con Euclideana\n")

    for label_number in range(1, labeled_array.max() + 1):
        labeled_region = (labeled_array == label_number)
        imagen_coloreada[labeled_region] = colores[label_number]

        area, perim, centroid = obj_areas[label_number -
                                          1], obj_perimeters[label_number - 1], obj_centroids[label_number - 1]
        object_label = object_labels[label_number - 1]
        object_name = object_names.get(
            object_label, 'Tipo de objeto desconocido')
        x, y = 0, 0
        text_position = (x, (y + (label_number * 15)))
        # cv2.putText(imagen_coloreada, f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        print(
            f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}')
        cv2.putText(imagen_coloreada, f'{label_number}', centroid, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)


    # Muestra la imagen
    plt.imshow(imagen_coloreada)
    plt.show()

def showImagesMahalanobis(original, cleaned_image, labeled_array, numero_imagen, obj_areas, obj_perimeters, obj_centroids, object_labels, object_names):
    labeled_array = labeled_array.astype(np.uint8)

    imagen_coloreada = np.zeros(
        (cleaned_image.shape[0], cleaned_image.shape[1], 3), dtype=np.uint8)
    colores = np.random.randint(
        50, 226, (labeled_array.max() + 1, 3), dtype=np.uint8)

    print("\nEntrenar imagenes con Mahalanobis\n")

    for label_number in range(1, labeled_array.max() + 1):
        labeled_region = (labeled_array == label_number)
        imagen_coloreada[labeled_region] = colores[label_number]

        area, perim, centroid = obj_areas[label_number -
                                          1], obj_perimeters[label_number - 1], obj_centroids[label_number - 1]
        object_label = object_labels[label_number - 1]
        object_name = object_names.get(
            object_label, 'Tipo de objeto desconocido')
        x, y = 0, 0
        text_position = (x, (y + (label_number * 15)))
        # cv2.putText(imagen_coloreada, f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        print(
            f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}')
        cv2.putText(imagen_coloreada, f'{label_number}', centroid, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)


    # Muestra la imagen
    plt.imshow(imagen_coloreada)
    plt.show()

def showImagesKNN(original, cleaned_image, labeled_array, numero_imagen, obj_areas, obj_perimeters, obj_centroids, object_labels, object_names):
    labeled_array = labeled_array.astype(np.uint8)

    imagen_coloreada = np.zeros(
        (cleaned_image.shape[0], cleaned_image.shape[1], 3), dtype=np.uint8)
    colores = np.random.randint(
        50, 226, (labeled_array.max() + 1, 3), dtype=np.uint8)

    print("\nEntrenar imagenes con KNN\n")

    for label_number in range(1, labeled_array.max() + 1):
        labeled_region = (labeled_array == label_number)
        imagen_coloreada[labeled_region] = colores[label_number]

        area, perim, centroid = obj_areas[label_number -
                                          1], obj_perimeters[label_number - 1], obj_centroids[label_number - 1]
        object_label = object_labels[label_number - 1]
        object_name = object_names.get(
            object_label, 'Tipo de objeto desconocido')
        x, y = 0, 0
        text_position = (x, (y + (label_number * 15)))
        # cv2.putText(imagen_coloreada, f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        print(
            f'Fig: {label_number}, Area: {area}, Perimeter: {perim}, Centroid: {centroid}, Objeto: {object_name}')
        cv2.putText(imagen_coloreada, f'{label_number}', centroid, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)


    # Muestra la imagen
    plt.imshow(imagen_coloreada)
    plt.show()

def train_all_classifiers(features, labels):
    classifiers = []

    # Entrenar clasificador de máxima probabilidad
    max_probability_classifier = train_max_probability_classifier(
        features, labels)
    classifiers.append(max_probability_classifier)

    # Entrenar clasificador euclidiano
    euclidean_classifier, _ = train_euclidean_classifier(features, labels)
    classifiers.append(euclidean_classifier)

    # Entrenar clasificador de Mahalanobis
    mahalanobis_features = np.array(features)
    mahalanobis_predictor = mahalanobis_classifier(
        mahalanobis_features, labels)
    classifiers.append(mahalanobis_predictor)

    # Entrenar clasificador KNN
    knn_classifier = train_knn_classifier(features, labels)
    classifiers.append(knn_classifier)

    # Agregar clasificador para objetos desconocidos
    unknown_labels = [0] * len(labels)  # Etiqueta 0 para objetos desconocidos
    unknown_classifier = train_max_probability_classifier(
        features, unknown_labels)
    classifiers.append(unknown_classifier)

    return classifiers

def tornillostrain():
    numero_imagen = 0
    imagen = loadImage(numero_imagen)

    if imagen is not None:
        img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_binaria = binarize(img_gris)
        cleaned_image = cleanImage(img_binaria)

        labeled_array, num_objects = countObjects(cleaned_image)

        obj_areas, obj_perimeters, obj_centroids = calculateAPC(cleaned_image)

        contours, _ = cv2.findContours(
            cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features = extract_features(contours, obj_centroids)

        labels = [1] * len(contours)

        c1 = train_all_classifiers(features, labels)

        object_labels_max_probability = c1[0].predict(features)
        object_labels_euclidean = [c1[1](f1, f2)
                                   for f1 in features for f2 in features]
        object_labels_mahalanobis = [c1[2](f) for f in features]
        object_labels_knn = c1[3].predict(features)

        labelfinal = object_labels_euclidean, object_labels_mahalanobis, object_labels_euclidean, object_labels_max_probability

        object_names = {
            1: "tornillo",
        }

        print("\nEntrenar tornillos")
    return c1, features, labelfinal

def rondanatrain():
    numero_imagen = 4
    imagen = loadImage(numero_imagen)

    if imagen is not None:
        img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_binaria = binarize(img_gris)
        cleaned_image = cleanImage(img_binaria)

        labeled_array, num_objects = countObjects(cleaned_image)

        obj_areas, obj_perimeters, obj_centroids = calculateAPC(cleaned_image)

        contours, _ = cv2.findContours(
            cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features = extract_features(contours, obj_centroids)
        labels = [1] * len(contours)

        c2 = train_all_classifiers(features, labels)

        object_labels_max_probability = c2[0].predict(features)
        object_labels_euclidean = [c2[1](f1, f2)
                                   for f1 in features for f2 in features]
        object_labels_mahalanobis = [c2[2](f) for f in features]
        object_labels_knn = c2[3].predict(features)

        labelfinal = object_labels_euclidean, object_labels_mahalanobis, object_labels_euclidean, object_labels_max_probability

        object_names = {
            1: "rondana",
        }

        print("\nEntrenar rondanas")

    
    return c2, features, labelfinal

def alcayatatrain():
    numero_imagen = 29
    imagen = loadImage(numero_imagen)

    if imagen is not None:
        img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_binaria = binarize(img_gris)
        cleaned_image = cleanImage(img_binaria)

        labeled_array, num_objects = countObjects(cleaned_image)

        obj_areas, obj_perimeters, obj_centroids = calculateAPC(cleaned_image)

        contours, _ = cv2.findContours(
            cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features = extract_features(contours, obj_centroids)
        labels = [1] * len(contours)

        c3 = train_all_classifiers(features, labels)

        object_labels_max_probability = c3[0].predict(features)
        object_labels_euclidean = [c3[1](f1, f2)
                                   for f1 in features for f2 in features]
        object_labels_mahalanobis = [c3[2](f) for f in features]
        object_labels_knn = c3[3].predict(features)

        labelfinal = object_labels_euclidean, object_labels_mahalanobis, object_labels_euclidean, object_labels_max_probability

        object_names = {
            1: "alcayata",
        }

        print("\nEntrenar alcayatas")

    return c3, features, labelfinal

def armellastrain():
    numero_imagen = 9
    imagen = loadImage(numero_imagen)

    if imagen is not None:
        img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_binaria = binarize(img_gris)
        cleaned_image = cleanImage(img_binaria)

        labeled_array, num_objects = countObjects(cleaned_image)

        obj_areas, obj_perimeters, obj_centroids = calculateAPC(cleaned_image)


        contours, _ = cv2.findContours(
            cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features = extract_features(contours, obj_centroids)
        labels = [1] * len(contours)

        c4 = train_all_classifiers(features, labels)

        object_labels_max_probability = c4[0].predict(features)
        object_labels_euclidean = [c4[1](f1, f2)
                                   for f1 in features for f2 in features]
        object_labels_mahalanobis = [c4[2](f) for f in features]
        object_labels_knn = c4[3].predict(features)

        labelfinal = object_labels_euclidean, object_labels_mahalanobis, object_labels_euclidean, object_labels_max_probability

        object_names = {
            1: "armellas",
        }

        print("\nEntrenar armellas")

    return c4, features, labelfinal

def colapatotrain():
    numero_imagen = 28
    imagen = loadImage(numero_imagen)

    if imagen is not None:
        img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_binaria = binarize(img_gris)
        cleaned_image = cleanImage(img_binaria)

        labeled_array, num_objects = countObjects(cleaned_image)

        obj_areas, obj_perimeters, obj_centroids = calculateAPC(cleaned_image)

        contours, _ = cv2.findContours(
            cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features = extract_features(contours, obj_centroids)
        labels = [1] * len(contours)

        c5 = train_all_classifiers(features, labels)

        object_labels_max_probability = c5[0].predict(features)
        object_labels_euclidean = [c5[1](f1, f2)
                                   for f1 in features for f2 in features]
        object_labels_mahalanobis = [c5[2](f) for f in features]
        object_labels_knn = c5[3].predict(features)

        labelfinal = object_labels_euclidean, object_labels_mahalanobis, object_labels_euclidean, object_labels_max_probability

        object_names = {
            1: "grapa cola de pato",
        }

        print("\nEntrenar Grapa cola de pato")

    return c5, features, labelfinal

def predict_by_feature_comparison(detected_features, reference_features, object_names):
    object_predictions = []

    # Para cada conjunto de características detectadas, encuentra el conjunto de referencia más cercano
    for detected in detected_features:
        min_distance = float('inf')
        predicted_label = None

        # Compara con cada conjunto de características de referencia
        for label, ref_features in reference_features.items():
            for ref in ref_features:
                dist = distance.euclidean(detected, ref)
                if dist < min_distance:
                    min_distance = dist
                    predicted_label = label

        # Asigna el nombre del objeto basado en la etiqueta con la menor distancia
        if predicted_label == 0:
            object_predictions.append('Desconocido')
        else:
            object_predictions.append(object_names.get(predicted_label, 'Desconocido'))

    return object_predictions

def classify_objects(object_names):

    # Entrenar clasificadores y obtener características y etiquetas para cada tipo de objeto
    c1, features_tornillo, labels_tornillo = tornillostrain()
    c2, features_rondana, labels_rondana = rondanatrain()
    c3, features_alcayata, labels_alcayata = alcayatatrain()
    c4, features_armellas, labels_armellas = armellastrain()
    c5, features_colapato, labels_colapato = colapatotrain()

    reference_features = {
        1: features_tornillo,
        2: features_rondana,
        3: features_alcayata,
        4: features_armellas,
        5: features_colapato,
    }
    
    while True:
        numero_imagen = int(input("¿Qué imagen deseas ver?: "))
        imagen = loadImage(numero_imagen)
        imagen2 = loadImage(numero_imagen)

        if imagen is not None:
            img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            img_binaria = binarize(img_gris)
            cleaned_image = cleanImage(img_binaria)

            labeled_array, num_objects = countObjects(cleaned_image)

            obj_areas, obj_perimeters, obj_centroids = calculateAPC(
                cleaned_image)

            print(f"Hay {num_objects} objetos en la imagen.")

            # Encuentra los contornos en la imagen
            contours, _ = cv2.findContours(
                cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Extrae características de los objetos en la imagen
            features = extract_features(contours, obj_centroids)

            # Predice los objetos usando los clasificadores de referencia
            object_predictions = predict_by_feature_comparison(
                features, reference_features, object_names)

            # Imprime las predicciones
            for prediction in object_predictions:
                print(prediction)

            repeticion = {}

            for prediction, count in repeticion.items():
                    print(f"{prediction}: {count} veces")

            for prediction in object_predictions:
                if prediction in repeticion:
                    repeticion[prediction] += 1
                else:
                    repeticion[prediction] = 1

            max_item = sorted(repeticion.items(), key=lambda x:x[1], reverse=True)
            print(dict(max_item))

            for i, centroid in enumerate(obj_centroids):
                cX, cY = centroid
                label = object_predictions[i]
                cv2.circle(imagen, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(imagen, label, (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Imagen Final", imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()   

            cv2.imshow("Imagen Final", imagen2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()   
 

            searchObjects_classified(max_item, object_names)


        continuar = input("¿Quieres ver otra imagen? (si/no): ")
        if continuar.lower() != 'si':
            print("¡Hasta luego!")
            break

def searchObjects(obj_type, object_names):

    # Entrenar clasificadores y obtener características y etiquetas para cada tipo de objeto
    c1, features_tornillo, labels_tornillo = tornillostrain()
    c2, features_rondana, labels_rondana = rondanatrain()
    c3, features_alcayata, labels_alcayata = alcayatatrain()
    c4, features_armellas, labels_armellas = armellastrain()
    c5, features_colapato, labels_colapato = colapatotrain()

    reference_features = {
        1: features_tornillo,
        2: features_rondana,
        3: features_alcayata,
        4: features_armellas,
        5: features_colapato,
    }

    obj_type = object_names[int(obj_type)]

    obj_images = []

    for i in range(113):
        imagen = loadImage(i)
        if imagen is not None:
                img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                img_binaria = binarize(img_gris)
                cleaned_image = cleanImage(img_binaria)

                labeled_array, num_objects = countObjects(cleaned_image)

                obj_areas, obj_perimeters, obj_centroids = calculateAPC(
                    cleaned_image)

                print(f"Hay {num_objects} objetos en la imagen.")

                contours, _ = cv2.findContours(
                    cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                features = extract_features(contours, obj_centroids)

                object_predictions = predict_by_feature_comparison(
                    features, reference_features, object_names)

                for prediction in object_predictions:
                    print(prediction)

                repeticion = {}

                for prediction in object_predictions:
                    if prediction in repeticion:
                        repeticion[prediction] += 1
                    else:
                        repeticion[prediction] = 1

                (max_item) = sorted(repeticion.items(), key=lambda x:x[1], reverse=True)
                print((max_item))

                if max_item[0][0] == obj_type:
                    obj_images.append(i)
                    
    print(obj_images)

    for images in obj_images:
        print(images)
        __path__ = 'bd_metales'
        img_path = f"{__path__}\\{images}.bmp"
        image = cv2.imread(img_path)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # for i, images in enumerate(obj_images):
    #     img_path = f"bd_metales/{images}.bmp"  # Usar '/' en lugar de '\\' para la ruta en sistemas Unix
    #     image = cv2.imread(img_path)

    #     # Determinar la posición del subgráfico actual
    #     row = i // 2
    #     col = i % 2

    #     # Mostrar la imagen en el subgráfico correspondiente
    #     axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     axs[row, col].set_title(images)

    #     # Ajustar el diseño para evitar solapamientos
    #     plt.tight_layout()
    #     plt.show()


def searchObjects_classified(max_item, object_names):

    c1, features_tornillo, labels_tornillo = tornillostrain()
    c2, features_rondana, labels_rondana = rondanatrain()
    c3, features_alcayata, labels_alcayata = alcayatatrain()
    c4, features_armellas, labels_armellas = armellastrain()
    c5, features_colapato, labels_colapato = colapatotrain()

    reference_features = {
        1: features_tornillo,
        2: features_rondana,
        3: features_alcayata,
        4: features_armellas,
        5: features_colapato,
    }

    obj_types = max_item

    obj_images = []

    diccionario_max_items = {}


    for i in range(114):
        print(i)
        imagen = loadImage(i)
        if imagen is not None:
                img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                img_binaria = binarize(img_gris)
                cleaned_image = cleanImage(img_binaria)

                labeled_array, num_objects = countObjects(cleaned_image)

                obj_areas, obj_perimeters, obj_centroids = calculateAPC(
                    cleaned_image)

                print(f"Hay {num_objects} objetos en la imagen.")

                contours, _ = cv2.findContours(
                    cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                features = extract_features(contours, obj_centroids)

                object_predictions = predict_by_feature_comparison(
                    features, reference_features, object_names)

                repeticion = {}

                for prediction in object_predictions:
                    if prediction in repeticion:
                        repeticion[prediction] += 1
                    else:
                        repeticion[prediction] = 1

                (max_item) = sorted(repeticion.items(), key=lambda x:x[1], reverse=True)
                # print(type(max_item))
                # print(max_item)

                # diccionario_max_items[imagen] = tuple(max_item)

                elementos_extraidos = [tupla[0] for tupla in obj_types]

                elementos_en_comun = [elemento for elemento in elementos_extraidos if elemento in object_predictions]

                if elementos_en_comun:    
                    obj_images.append(i)
    
    print(diccionario_max_items)
                
    for images in obj_images:
        print(images)
        __path__ = 'bd_metales'
        img_path = f"{__path__}\\{images}.bmp"
        image = cv2.imread(img_path)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    # for i, images in enumerate(obj_images):
    #     img_path = f"bd_metales/{images}.bmp"  # Usar '/' en lugar de '\\' para la ruta en sistemas Unix
    #     image = cv2.imread(img_path)

    #     # Determinar la posición del subgráfico actual
    #     row = i // 2
    #     col = i % 2

    #     # Mostrar la imagen en el subgráfico correspondiente
    #     axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     axs[row, col].set_title(images)

    #     # Ajustar el diseño para evitar solapamientos
    #     plt.tight_layout()
    #     plt.show()

def main():

    object_names = {
        1: 'Tornillo',
        2: 'Rondana',
        3: 'Alcayata',
        4: 'Armella',
        5: 'Colapato',
    }
    
    while True:
        print("Que deseas realizar?")
        opcion = input("1. Obtener imagenes de ciertos elementos, 2. Clasificar elementos en una imagen, 3. Comparar con imagen desconocida, Otro: Salir  ")
        if opcion == '1':
            obj_type = input("1. Tornillo, 2. Rondana, 3. Alcayata, 4. Armella, 5. Colapato:  ")
            searchObjects(obj_type, object_names)
        if opcion == '2':
            classify_objects(object_names)
        else:
            print("Bye!")
            break

if __name__ == "__main__":
    main()