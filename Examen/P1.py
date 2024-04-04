import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def plot_labeled_image(image, connectivity):
    labeled_image, num_features = label(image, structure=connectivity)
    plt.figure(figsize=(10, 5))

    cmap = plt.get_cmap('tab20b', num_features + 1)
    plt.imshow(labeled_image, cmap=cmap)
    plt.colorbar()
    plt.title(f'Número de objetos detectados: {num_features}')
    plt.show()

# Define la imagen
image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
                  [0, 1, 1, 1, 1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
                  [0, 1, 1, 1, 1, 0, 0, 0, 0, 0,0,0,1,1,1,1,0,0],
                  [0, 1, 1, 1, 0, 0, 0, 0, 0, 0,0,0,1,1,1,1,0,0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0,0,0,1,1,1,1,0,0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 1,1,1,1,1,1,1,0,0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0,0,1,1,1,1,1,0,0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0,0,1,1,1,1,1,1,0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0,0,0,0,1,1,1,0,0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]])

# Conectividad 4
plot_labeled_image(image, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# Conectividad 8
plot_labeled_image(image, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])