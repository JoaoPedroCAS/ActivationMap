import torchvision.models as models
import torch.nn as nn


model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-9])
model.add_module("GAP", nn.AdaptiveAvgPool2d((1, 1)))
# Imprimir todas as camadas e seus detalhes
for name, module in model.named_children():
    print(f"Camada: {name}, Tipo: {module.__class__.__name__}")
    
    # Inspecionando os parâmetros da camada
    for param_name, param in module.named_parameters():
        print(f"  Parâmetro: {param_name} - Forma: {param.shape}")
        
    # Se a camada tem um método forward, você pode verificar sua configuração
    if hasattr(module, 'forward'):
        print(f"  Método Forward: {module.forward}")
    print("-" * 50)  # Separador entre as camadas
