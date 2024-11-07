import os
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Substituição do SVM pelo LDA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

# Define o caminho para o seu dataset
dataset_path = '/home/joao.p.c.a.sa/PreProjeto/Dataset/DTD/dtd/dtd/images'

# Carrega o modelo ResNet50 e remove a camada de classificação
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.model.children())[:-2])  # Remove a camada de classificação
    
    def forward(self, x):
        return self.features(x)

# Define transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder):
    print("Carregando imagens da pasta...")
    images = []
    labels = []
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Erro ao carregar a imagem {img_path}: {e}")
    print(f"Carregadas {len(images)} imagens.")
    return images, labels

# Carregar o dataset
print("Iniciando o carregamento do dataset...")
images, labels = load_images_from_folder(dataset_path)
print(f"Carregadas {len(images)} imagens e {len(np.unique(labels))} classes.")

# Converter lista de tensores em um único tensor
images_tensor = torch.stack(images)
print("Convertida a lista de imagens para tensor.")

# Extrair características usando ResNet50
print("Iniciando extração de características usando ResNet50...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50FeatureExtractor().to(device)
model.eval()

features = []
for i in range(len(images_tensor)):
    img = images_tensor[i].unsqueeze(0).to(device)
    feature = model(img).cpu().detach().numpy()  # Corrigido para detach
    features.append(feature.flatten())
print(f"Extraídas características para {len(features)} imagens.")

features = np.array(features)
labels = np.array(labels)
print(f"Forma das características: {features.shape}, Forma dos labels: {labels.shape}")

# Executar LDA e validação cruzada
print("Inicializando o modelo LDA...")
lda = LinearDiscriminantAnalysis()  # Modelo LDA
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
accuracy_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

print("Iniciando validação cruzada...")
for fold, (train_index, test_index) in enumerate(cv.split(features, labels), 1):
    print(f"Processando fold {fold}...")
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Treinar o modelo
    print("Treinando o modelo LDA...")
    lda.fit(X_train, y_train)

    # Predizer e avaliar o modelo
    print("Realizando previsões...")
    y_pred = lda.predict(X_test)
    print("Gerando report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = np.mean([report[str(i)]['f1-score'] for i in range(len(np.unique(labels))) if str(i) in report])
    recall = np.mean([report[str(i)]['recall'] for i in range(len(np.unique(labels))) if str(i) in report])
    precision = np.mean([report[str(i)]['precision'] for i in range(len(np.unique(labels))) if str(i) in report])

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

print(f"Validação cruzada concluída. Calculando métricas...")

# Imprimir as métricas
print(f"Accuracy: {np.mean(accuracy_scores)} +- {np.std(accuracy_scores)}")
print(f"F1-Score: {np.mean(f1_scores)} +- {np.std(f1_scores)}")
print(f"Recall: {np.mean(recall_scores)} +- {np.std(recall_scores)}")
print(f"Precision: {np.mean(precision_scores)} +- {np.std(precision_scores)}")

