import torchvision.models as models

model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
for name, module in model.named_children():
	print(f"Camada: {name}, Tipo:: {module.__class__.__name__}")
