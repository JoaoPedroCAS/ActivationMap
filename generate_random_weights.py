import numpy as np
import os

# Verifica e cria a pasta se não existir
os.makedirs('random_weights', exist_ok=True)

# Dimensões do tensor
shape = (64, 3, 7, 7)

# Gera e salva 200 arquivos
for i in range(1, 201):
    # Gera valores aleatórios entre -2 e 2 no formato especificado
    valores_aleatorios = np.random.uniform(-1, 1, shape).flatten()
    
    # Define o nome do arquivo com número sequencial
    nome_arquivo = f'random_weights/random_weights_{i}.txt'
    
    # Salva os valores em uma única linha no arquivo
    with open(nome_arquivo, 'w') as f:
        f.write(','.join(map(str, valores_aleatorios)))
