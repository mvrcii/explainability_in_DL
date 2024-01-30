import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import transforms

imagenet_labels = dict(enumerate(open('vit/imagenet_1k_labels.txt')))

# Load the model
model = torch.load("vit/model.pth")
model.eval()

img = Image.open('vit/dog.jpg')
plt.imshow(img)
plt.show()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
x = transform(img).unsqueeze(0)

logits = model(x)
probs = torch.nn.functional.softmax(logits, dim=-1)

k = 10
top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob_ = prob_.item()
    clazz = imagenet_labels[ix].strip()
    print(f"{i + 1:2}: {clazz:<45} --- {prob_:.4f}")
