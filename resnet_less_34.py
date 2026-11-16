from PIL import Image
from torchvision import models

resnet_weights = models.ResNet50_Weights.DEFAULT
print(resnet_weights.meta['categories'][:10])
cats = resnet_weights.meta['categories']
transforms = resnet_weights.transforms()
model = models.resnet50(weights=resnet_weights)
# print(model)

img = Image.open('images/IMG_9223.jpg').convert('RGB')
img = transforms(img).unsqueeze(0) # (1, 3, 224, 224)

model.eval()
p = model(img).squeeze()
res = p.softmax(dim=0).sort(descending=True)

for s, i in zip(res[0][:5], res[1][:5]):
    print(f"{cats[i]}: {s:.4f}")