close all
minimo = input('Ingrese el valor mínimo: ');
maximo = input('Ingrese el valor máximo: ');

numeros_primos = [];
for num = minimo:maximo
    if isprime(num)
        numeros_primos = [numeros_primos, num];
    end
end

numeros_no_primos = setdiff(minimo:maximo, numeros_primos);

figure(1);
scatter(numeros_primos, zeros(size(numeros_primos)), 'b', 'filled');
hold on;
title('Números Primos en un Rango');
xlabel('Número');
ylim([-1, 1]);

figure(2);
scatter(numeros_no_primos, zeros(size(numeros_no_primos)), 'r', 'filled');
hold on;
title('Números no Primos en un Rango');
xlabel('Número');
ylim([-1, 1]);

vector = input('Ingrese un vector: ');

% Verificar si el vector está dentro del rango
if all(vector >= minimo) && all(vector <= maximo)
    disp('Analizando');
    figure(3)
    scatter(numeros_primos, zeros(size(numeros_primos)), 'b', 'filled');
    hold on
    scatter(numeros_no_primos, zeros(size(numeros_no_primos)), 'r', 'filled');
    hold on
    scatter(vector, 0, 'g', 'filled');
    
    hold off;
    
    clasificarConDistanciaEuclidiana(numeros_primos, numeros_no_primos, vector);
    hold on;
    legend('Números Primos', 'Números no Primos', "Vector");
else
    disp('El vector no pertenece a ninguna clase.');
end



function clasificarConDistanciaEuclidiana(numeros_primos, numeros_no_primos, vector)

    distancia_primera_clase = norm(vector - numeros_primos);
    distancia_segunda_clase = norm(vector - numeros_no_primos);

    if distancia_primera_clase < distancia_segunda_clase
        fprintf('El vector pertenece a la Clase 1 (Números Primos).\n');
    else
        fprintf('El vector pertenece a la Clase 2 (Números no Primos).\n');
        
    end
end