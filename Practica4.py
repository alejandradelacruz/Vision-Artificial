import cv2
import numpy as np

image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
                  [0, 1, 1, 1, 1, 0, 0, 0, 0, 0,0,0,1,1,1,1,0,0],
                  [0, 1, 1, 1, 0, 0, 0, 0, 0, 0,0,0,1,1,1,1,0,0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0,0,0,1,1,1,1,0,0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 1,1,1,1,1,1,1,0,0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0,0,1,1,1,1,1,0,0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0,0,1,1,1,1,1,1,0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0,0,0,0,1,1,1,0,0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]])

def cargar_imagen_desde_array(imagen_array):
    return imagen_array

def binarizar_imagen(img):
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return binary_img

def mostrar_imagenes(original, procesadas, nombres_metodos):
    cv2.imshow("Original", original)

    for i, (img, metodo) in enumerate(zip(procesadas, nombres_metodos), 1):
        cv2.imshow(f"{metodo}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def obtener_tamano_kernel():
    tamano_kernel = int(input("Ingrese el tamaño del kernel: "))
    if tamano_kernel%2 == 0:
        tamano_kernel=tamano_kernel+1
        return (tamano_kernel, tamano_kernel)
    else:
        return (tamano_kernel, tamano_kernel)

def dilatar(img, tamano_kernel):
    kernels = {
        "Rectangular": np.ones(tamano_kernel, np.uint8),
        "Cruz": cv2.getStructuringElement(cv2.MORPH_CROSS, tamano_kernel),
        "Circular": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamano_kernel),
        "Forma L": np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], np.uint8)
    }

    binary_img = binarizar_imagen(img)
    imagenes_dilatadas = [dilate(binary_img, kernels[kernel]) for kernel in kernels]
    mostrar_imagenes(img, imagenes_dilatadas, list(kernels.keys()))

def dilate(binary_img, kernel):
    rows, cols = binary_img.shape
    output_img = np.zeros((rows, cols), dtype=np.uint8)

    kernel_rows, kernel_cols = kernel.shape
    k_center = (kernel_rows // 2, kernel_cols // 2)

    for i in range(k_center[0], rows - k_center[0]):
        for j in range(k_center[1], cols - k_center[1]):
            roi = binary_img[i - k_center[0]:i + k_center[0] + 1, j - k_center[1]:j + k_center[1] + 1]
            if np.sum(roi & kernel) > 0:
                output_img[i, j] = 255
    return output_img

def erosionar(img, tamano_kernel):
    kernels = {
        "Rectangular": np.ones(tamano_kernel, np.uint8),
        "Cruz": cv2.getStructuringElement(cv2.MORPH_CROSS, tamano_kernel),
        "Circular": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamano_kernel),
        "Forma L": np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], np.uint8)
    }

    binary_img = binarizar_imagen(img)
    imagenes_erosionadas = [erode(binary_img, kernels[kernel]) for kernel in kernels]
    mostrar_imagenes(img, imagenes_erosionadas, list(kernels.keys()))

def erode(binary_img, kernel):
    rows, cols = binary_img.shape
    output_img = np.zeros((rows, cols), dtype=np.uint8)

    kernel_rows, kernel_cols = kernel.shape
    k_center = (kernel_rows // 2, kernel_cols // 2)

    for i in range(k_center[0], rows - k_center[0]):
        for j in range(k_center[1], cols - k_center[1]):
            roi = binary_img[i - k_center[0]:i + k_center[0] + 1, j - k_center[1]:j + k_center[1] + 1]
            if np.sum(roi & kernel) == np.sum(kernel):  
                output_img[i, j] = 255
    return output_img
   
def open_image(binary_img, kernel):
    eroded_img = erode(binary_img, kernel)
    opened_img = dilate(eroded_img, kernel)
    cv2.imshow("Opened Image", opened_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return opened_img

def close_image(binary_img, kernel):
    dilated_img = dilate(binary_img, kernel)
    closed_img = erode(dilated_img, kernel)
    cv2.imshow("Closed Image", closed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return closed_img
    
def obtener_contornos(img, tamano_kernel):
    binary_img = binarizar_imagen(img)
    kernels = {
        "Rectangular": np.ones(tamano_kernel, np.uint8),
        "Cruz": cv2.getStructuringElement(cv2.MORPH_CROSS, tamano_kernel),
        "Circular": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamano_kernel),
        "Forma L": np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], np.uint8)
    }

    imagenes_contornos = []
    nombres_kernels = []

    for nombre, kernel in kernels.items():
        img_dilatada = dilate(binary_img, kernel)
        img_erosionada = erode(binary_img, kernel)
        #contornos = img_dilatada - img_erosionada
        cierre = img_dilatada - img_erosionada
        apertura = img_erosionada - img_dilatada
        #cv2.imshow("Apertura", apertura)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        contornos = cierre - apertura
        
        imagenes_contornos.append(contornos)
        nombres_kernels.append(f"Contornos {nombre}")

    mostrar_imagenes(img, imagenes_contornos, nombres_kernels)

def main():
    while True:
        # Cambiar la entrada de la imagen
        imagen_original = cargar_imagen_desde_array(image)

        if imagen_original is not None:
            print("Seleccione una opción:")
            print("1. Dilatar")
            print("2. Erosionar")
            print("3. Contornos")

            opcion = int(input("Ingrese el número de la opción: "))
            tamano_kernel = obtener_tamano_kernel()

            if opcion == 1:
                dilatar(imagen_original, tamano_kernel)
            elif opcion == 2:
                erosionar(imagen_original, tamano_kernel)
            elif opcion == 3:
                contornos = obtener_contornos(imagen_original, tamano_kernel)
                cv2.imshow("Contornos Cierre", contornos)
                cv2.waitKey(0)
                cv2.destroyAllWindows()  
            else:
                print("Opción no válida.")
            op2 = input("¿Desea ver otra imagen? (si/no): ").lower()
            if op2 != 'si':
                break
        else:
            print("No se puede procesar la imagen.")

if __name__ == "__main__":
    main()