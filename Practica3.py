import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, grey_dilation, grey_erosion

def plot_labeled_image(image, connectivity):
    labeled_image, num_features = label(image, structure=connectivity)
    plt.figure(figsize=(10, 7))

    cmap = plt.get_cmap('tab20b', num_features + 1)
    plt.imshow(labeled_image, cmap=cmap)
    plt.colorbar()
    plt.title(f'Número de objetos detectados: {num_features}')
    plt.show()

def apply_dilation_erosion(image):
    struct_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    struct_square = np.ones((3, 3))
    struct_ellipse = np.array([[0,1, 1, 1,0], [1, 1, 1,1,1],[1, 1, 1,1,1],[1, 1, 1,1,1], [0,1, 1, 1,0]])

    dilated_cross = grey_dilation(image, structure=struct_cross)
    eroded_cross = grey_erosion(image, structure=struct_cross)

    dilated_square = grey_dilation(image, structure=struct_square)
    eroded_square = grey_erosion(image, structure=struct_square)

    dilated_ellipse = grey_dilation(image, structure=struct_ellipse)
    eroded_ellipse = grey_erosion(image, structure=struct_ellipse)

    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(eroded_ellipse, cmap='gray')
    plt.title('Erosion - Circulo')

    plt.subplot(232)
    plt.imshow(dilated_cross, cmap='gray')
    plt.title('Dilatación - Cruz')

    plt.subplot(233)
    plt.imshow(eroded_cross, cmap='gray')
    plt.title('Erosión - Cruz')

    plt.subplot(234)
    plt.imshow(dilated_square, cmap='gray')
    plt.title('Dilatación - Cuadrado')

    plt.subplot(235)
    plt.imshow(eroded_square, cmap='gray')
    plt.title('Erosión - Cuadrado')

    plt.subplot(236)
    plt.imshow(dilated_ellipse, cmap='gray')
    plt.title('Dilatación - Elipse')

    # plt.subplot(237)
    # plt.imshow(eroded_ellipse, cmap='gray')
    # plt.title('Erosión - Elipse')

    plt.tight_layout()
    plt.show()

# Define la imagen en escala de grises
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

apply_dilation_erosion(image)