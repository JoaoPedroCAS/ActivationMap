import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

# Define o caminho para o seu dataset e para salvar as imagens dos mapas de ativação
dataset_path = '/home/joao.p.c.a.sa/PreProjeto/Datasets'
output_dir = '/home/joao.p.c.a.sa/PreProjeto/Code/activation_maps_layer_0'
os.makedirs(output_dir, exist_ok=True)
print("Diretorios Criados")

# Carrega o modelo ResNet50 e remove a camada de classificação
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-9])
        self.model.add_module("GAP", nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        return self.model(x)

# Define transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Transformações Definidas")
# Função para carregar imagens
def load_images_from_folder(folder):
    images = []
    labels = []
    original_images = []  # Para armazenar as imagens originais
    filenames = []
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    original_images.append(img.copy())  # Salvar a imagem original para visualização
                    img = transform(img)
                    images.append(img)
                    labels.append(label)
                    filenames.append(filename)
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
    return images, labels, original_images, filenames

# Carregar o dataset
images, labels, original_images, filenames = load_images_from_folder(dataset_path)
images_tensor = torch.stack(images)
print("Imagens Carregadas")

# Adicionar hook na camada X
def hook_fn(module, input, output):
    # Aqui podemos salvar ou processar as ativações, que são as saídas da camada
    return output

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50FeatureExtractor().to(device)
model.eval()
print("Modelo Avaliado")
# Registra o hook na camada X
layer_to_hook = model.model[0]
hook = layer_to_hook.register_forward_hook(hook_fn)
print("Hook feito")

# Inicializar lista para armazenar as características
features = []

# Extração de características e geração de mapas de ativação
target_filenames = [f'c{str(i+1).zfill(2)}_001_a_w01.png' for i in range(20)]
for i in range(len(images_tensor)):
    img = images_tensor[i].unsqueeze(0).to(device)

    with torch.no_grad():
        activation = model(img)  # Extrai as ativações da camada
        feature = activation.cpu().numpy().flatten()
        features.append(feature)

    # Gera o mapa de ativação
    if filenames[i] in target_filenames:
        activation_map = activation.squeeze(0).mean(dim=0).cpu().numpy()  # Calcula a média dos mapas de ativação

        # Normaliza o mapa de ativação para [0, 1] para visualização
        denominator = activation_map.max() - activation_map.min()
        if denominator == 0:
            activation_map = np.zeros_like(activation_map)  # Ou use um valor constante, como 0.5
        else:
            activation_map = (activation_map - activation_map.min()) / denominator

        # Exibir a imagem original e o mapa de ativação sobreposto
        # Plotar a imagem e o CAM
        img = img.cpu().squeeze(0).permute(1, 2, 0)
        activation_map_path = os.path.join(output_dir, f'{filenames[i]}_activation_map.png')
        plt.subplot(1, 2, 1)
        plt.imshow(img, alpha=0.8)
        plt.imshow(activation_map, cmap="jet", alpha=0.5)
        plt.title(f"CAM para {filenames[i]}")
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.savefig(activation_map_path)
        plt.close()

print("Mapas gerados")
