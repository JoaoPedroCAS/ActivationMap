import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np

# Caminho para o diretório onde o DTD está armazenado
dataset_dir = '/home/joao.p.c.a.sa/PreProjeto/Dataset/DTD/dtd/dtd/images'  # Altere para o caminho correto

# Definir as transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregar o dataset DTD (assumindo que você tem a estrutura correta de pastas)
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Criar um DataLoader para carregar o dataset
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Carregar o modelo pré-treinado (ResNet18)
model = models.resnet18(pretrained=True)
model.eval()  # Coloca o modelo em modo de avaliação

# Função para capturar as ativações de uma camada
def get_activation_map(model, input_image, target_layer):
    activation = {}

    def hook_fn(module, input, output):
        activation['value'] = output

    # Registrar um hook na camada desejada
    target_layer.register_forward_hook(hook_fn)

    # Passar a imagem pela rede
    output = model(input_image)

    # Retornar a ativação
    return activation['value']

# Pegar uma imagem do DataLoader
input_image, _ = next(iter(data_loader))  # Pega uma imagem do DataLoader

# Pegar a ativação de uma camada convolucional, como a primeira camada convolucional
target_layer = model.conv1

# Obter a ativação
activation_map = get_activation_map(model, input_image, target_layer)

# Converter a ativação para um formato visualizável
activation_map = activation_map.squeeze(0)  # Remover a dimensão do batch
activation_map = activation_map.mean(dim=0)  # Tirar a média sobre os canais (se necessário)

# Normalizar a ativação para que fique na faixa [0, 1] para visualização
activation_map = F.relu(activation_map)
activation_map = activation_map - activation_map.min()
activation_map = activation_map / activation_map.max()

# Convertendo a imagem original para numpy para plotar
input_image = input_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
input_image = np.clip(input_image, 0, 1)

# Criar subplots para mostrar a imagem original e a imagem com o mapa de ativação sobreposto
plt.subplot(1,2,1)
plt.imshow(input_image, alpha=0.8)
plt.imshow(activation_map.detach().cpu().numpy(), cmap='jet',alpha=0.5)
plt.subplot(1,2,2)
plt.imshow(input_image)
plt.savefig('/home/joao.p.c.a.sa/PreProjeto/Code/Image.jpg')
