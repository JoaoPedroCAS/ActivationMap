import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np

# Carregar modelo com a estrutura desejada
model = models.resnet50(weights=None)  # Não carrega pesos pretreinados
model = nn.Sequential(*list(model.children())[:-9])
model.add_module("GAP", nn.AdaptiveAvgPool2d((1, 1)))

# Função para carregar pesos de um arquivo txt
def carregar_pesos_personalizados(model, arquivo_pesos):
    with open(arquivo_pesos, 'r') as f:
        # Lê todos os valores como uma única linha
        valores = f.read().strip().split(',')
        valores = np.array([float(valor) for valor in valores])
    
    indice = 0
    for name, module in model.named_children():
        for param_name, param in module.named_parameters():
            # Verifica o tamanho esperado para a camada atual
            tamanho_param = param.numel()
            # Extrai o número exato de valores e converte para tensor
            pesos_tensor = torch.tensor(valores[indice:indice + tamanho_param]).view(param.shape)
            # Atualiza os pesos da camada
            param.data = pesos_tensor
            indice += tamanho_param

# Caminho do arquivo de pesos personalizados
arquivo_pesos = 'random_weights/random_weights_1.txt'
carregar_pesos_personalizados(model, arquivo_pesos)

# Verificar se os pesos foram carregados corretamente
for name, module in model.named_children():
    print(f"Camada: {name}, Tipo: {module.__class__.__name__}")
    for param_name, param in module.named_parameters():
        print(f"  {param_name}: {param.data[:5]}")  # Exibe os primeiros 5 elementos do tensor
