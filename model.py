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
print(layer_to_hook)
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
        activation_map = np.clip(activation_map, 0, np.percentile(activation_map, 95))  # Ignora 5% das ativações mais baixas
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())

        # Redimensiona o mapa de ativação para a resolução da imagem original
        activation_map_resized = Image.fromarray(activation_map)
        activation_map_resized = activation_map_resized.resize((224, 224), Image.BILINEAR)
        activation_map_resized = np.array(activation_map_resized)

        # Exibir a imagem original e o mapa de ativação sobreposto
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Subplot 1: Imagem original
        axes[0].imshow(original_images[i])
        axes[0].axis('off')
        axes[0].set_title('Imagem Original')

        # Subplot 2: Imagem com mapa de ativação sobreposto
        alpha = 0.5  # Ajuste o valor de alpha (0.0 para completamente transparente, 1.0 para completamente opaco)
        axes[1].imshow(original_images[i], alpha=1 - alpha)  # Imagem com opacidade ajustada
        axes[1].imshow(activation_map_resized, cmap='jet', alpha=alpha)  # Mapa de ativação sobreposto com alpha ajustado
        axes[1].axis('off')
        axes[1].set_title('Imagem com Mapa de Ativação')

        # Salvar a imagem do mapa de ativação
        activation_map_path = os.path.join(output_dir, f'{filenames[i]}_activation_map.png')
        plt.savefig(activation_map_path)
        plt.close(fig)

print("Mapas gerados")
