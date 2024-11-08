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
dataset_path = '/home/joao.p.c.a.sa/PreProjeto/Datasets'


#Carrega o modelo ResNet50 e remove a camada de classificação
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
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

#Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50FeatureExtractor().to(device)
model.eval()
print("Modelo Avaliado")

# Inicializar lista para armazenar as características
features = []

# Extração de características e geração de mapas de ativação
for i in range(len(images_tensor)):
    img = images_tensor[i].unsqueeze(0).to(device)

    with torch.no_grad():
        activation = model(img)  # Extrai as ativações da camada
        feature = activation.cpu().numpy().flatten()
        features.append(feature)

print("Features extraídas")
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
    print("Iniciando o LDA com RSKF")
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print("Treino e teste separado")
    # Treinar o modelo LDA
    lda.fit(X_train, y_train)
    print("Fit do modelo realizado")
    # Realizar previsões e avaliar o modelo
    y_pred = lda.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = np.mean([report[str(i)]['f1-score'] for i in range(len(np.unique(labels))) if str(i) in report])
    recall = np.mean([report[str(i)]['recall'] for i in range(len(np.unique(labels))) if str(i) in report])
    precision = np.mean([report[str(i)]['precision'] for i in range(len(np.unique(labels))) if str(i) in report])

    print(f"Acurácia: {accuracy}\nF1-Score: {f1}\nRecall: {recall}\nPrecision: {precision}\n----------")
    
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
