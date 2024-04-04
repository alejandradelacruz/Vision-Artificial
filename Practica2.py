import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

# Definir la imagen
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Definir los kernels para erosión y dilatación
kernel_erosion = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

kernel_dilation = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

# Aplicar erosión y dilatación
eroded_image = morphology.binary_erosion(image, structure=kernel_erosion)
dilated_image = morphology.binary_dilation(image, structure=kernel_dilation)

# Calcular la imagen de bordes
edge_image = np.logical_xor(dilated_image, eroded_image)

# Invertir los colores en la imagen de bordes
inverted_edge_image = ~edge_image

# Mostrar solo la detección de bordes
plt.figure(figsize=(8, 4))
plt.imshow(inverted_edge_image, cmap='gray')
plt.title('Detección de Bordes Cruz')
plt.axis('off')
plt.show()