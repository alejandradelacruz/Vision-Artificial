import matplotlib.pyplot as plt

class NumerosPrimos:
    def es_primo(self, num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def encontrar_primos_en_intervalo(self, rango_min, rango_max):
        primos = [num for num in range(rango_min, rango_max + 1) if self.es_primo(num)]
        return primos

# Creamos una instancia de la clase NumerosPrimos
primos = NumerosPrimos()

# Clase 1: Números primos en un intervalo [min, max]
min_intervalo = int(input("Ingresa el valor mínimo del intervalo: "))
max_intervalo = int(input("Ingresa el valor máximo del intervalo: "))
primos_clase_1 = primos.encontrar_primos_en_intervalo(min_intervalo, max_intervalo)

# Clase 2: Números primos impares en un intervalo [min, max]
impares_clase_2 = primos.encontrar_primos_en_intervalo(min_intervalo, max_intervalo)
impares_clase_2 = [num for num in impares_clase_2 if num % 2 != 0]

# Imprimimos las listas y su clasificación
print("Clase 1: Números primos en [min, max]")
print("Lista de primos:", primos_clase_1)

print("\nClase 2: Números primos impares en [min, max]")
print("Lista de impares:", impares_clase_2)

# Graficamos los resultados
plt.scatter(primos_clase_1, [1] * len(primos_clase_1), label='Clase 1: Primos en [min, max]', marker='o')
plt.scatter(impares_clase_2, [2] * len(impares_clase_2), label='Clase 2: Impares en [min, max]', marker='x')

# Configuramos la leyenda y etiquetas
plt.legend()
plt.yticks([1, 2], ['Clase 1', 'Clase 2'])
plt.title('Gráfico de Números Primos en Intervalo')

# Mostramos el gráfico
plt.show()
