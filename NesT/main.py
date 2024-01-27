import json
from io import BytesIO

import requests
import timm
import torch
from PIL import Image
from timm.data import create_transform


def load_imagenet_labels(filepath):
    with open(filepath, 'r') as f:
        labels = json.load(f)
    return labels


def load_and_transform_image(url, transform):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    return tensor


if __name__ == '__main__':
    imagenet_labels = load_imagenet_labels('imagenet_1k_labels.json')

    # Load the model
    model_name = 'nest_base'
    model = timm.create_model('nest_base_jx.goog_in1k', pretrained=True)
    model = model.eval()
    print(model.num_classes)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Load and transform an image
    image_url = 'https://picsum.photos/id/237/200/300'  # Replace with your image URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    # img = Image.open(urlopen(
    #     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    # ))

    img_tensor = transforms(img).unsqueeze(0)  # unsqueeze single image into batch of 1

    with torch.no_grad():
        logits = model(img_tensor)

    # Top-1 Prediction
    top1_prob, top1_cls = torch.max(logits.softmax(dim=1), dim=1)
    top1_label = imagenet_labels[top1_cls[0]]
    print(f'Top-1 Prediction: {top1_label} with probability {top1_prob[0] * 100:.2f}%')

    # Top-5 Predictions
    top5_probs, top5_classes = torch.topk(logits.softmax(dim=1), 5)
    print("Top-5 Predictions:")
    for i in range(5):
        label = imagenet_labels[top5_classes[0][i]]
        prob = top5_probs[0][i].item()
        print(f"{label}: {prob * 100:.2f}%")

    # cls = logits.argmax(axis=-1)
    # cls_name = imagenet_labels[cls]
    # prob = torch.nn.functional.softmax(logits[0], dim=0).max(dim=-1)
    # print(f'ImageNet class id: {cls[0]}, class name: {cls_name}, prob: {prob[0]}')
    #
    # top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
    # print(top5_probabilities, top5_class_indices)
