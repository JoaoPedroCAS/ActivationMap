import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

# Define o caminho para o seu dataset e para salvar as imagens dos mapas de ativação
dataset_path = '/home/joao.p.c.a.sa/PreProjeto/Dataset/DTD/dtd/dtd/images'
output_dir = '/home/joao.p.c.a.sa/PreProjeto/Code/activation_maps'  # Diretório para salvar as imagens dos mapas de ativação
os.makedirs(output_dir, exist_ok=True)
print("Diretorios Criados")

# Carrega o modelo ResNet50 e remove a camada de classificação
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove a camada de classificação
    
    def forward(self, x):
        return self.model(x)

# Definir transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Transformações Definidas")
# Carregar e pré-processar as imagens
def load_images_from_folder(folder):
    images = []
    labels = []
    original_images = []  # Para armazenar as imagens originais
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
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
    return images, labels, original_images

# Carregar o dataset
images, labels, original_images = load_images_from_folder(dataset_path)
images_tensor = torch.stack(images)
print("Imagens Carregadas")
# Extrair características e gerar mapas de ativação
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50FeatureExtractor().to(device)
model.eval()
print("Modelo Avaliado")
features = []
for i in range(len(images_tensor)):
    img = images_tensor[i].unsqueeze(0).to(device)
    
    with torch.no_grad():
        activation = model(img)  # Extraí a ativação da última camada convolucional
        feature = activation.cpu().numpy().flatten()
        features.append(feature)
    print("Features Extraidos")
    # Gera o mapa de ativação
    activation_map = activation.squeeze(0).mean(dim=0).cpu().numpy()  # Calcula a média dos mapas de ativação
    
    # Normaliza o mapa de ativação para [0, 1] para visualização
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    
    # Exibir a imagem original e o mapa de ativação sobreposto
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Subplot 1: Imagem original
    axes[0].imshow(original_images[i])
    axes[0].axis('off')
    axes[0].set_title('Imagem Original')
    
    # Subplot 2: Imagem com mapa de ativação sobreposto
    axes[1].imshow(original_images[i])
    axes[1].imshow(activation_map, cmap='jet', alpha=0.5)  # Sobrepõe o mapa de ativação
    axes[1].axis('off')
    axes[1].set_title('Imagem com Mapa de Ativação')

    # Salvar a imagem do mapa de ativação
    activation_map_path = os.path.join(output_dir, f'activation_map_{i}.png')
    plt.savefig(activation_map_path)
    plt.close(fig)
    print("Mapa Gerado")
# Converter lista de características e rótulos em arrays
features = np.array(features)
labels = np.array(labels)

# Executar LDA e validação cruzada
lda = LinearDiscriminantAnalysis()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
accuracy_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

for fold, (train_index, test_index) in enumerate(cv.split(features, labels), 1):
    print("Executando LDA com RSKF")
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Treinar o modelo LDA
    lda.fit(X_train, y_train)

    # Realizar previsões e avaliar o modelo
    y_pred = lda.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = np.mean([report[str(i)]['f1-score'] for i in range(len(np.unique(labels))) if str(i) in report])
    recall = np.mean([report[str(i)]['recall'] for i in range(len(np.unique(labels))) if str(i) in report])
    precision = np.mean([report[str(i)]['precision'] for i in range(len(np.unique(labels))) if str(i) in report])

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

# Imprimir as métricas finais da validação cruzada
print("Resultados da Validação Cruzada:")
print(f"Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
