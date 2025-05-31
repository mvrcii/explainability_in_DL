import json
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import timm
import torch
from PIL import Image
from openai import OpenAI
from timm.data import create_transform
from timm.models import Nest

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
    imagenet_labels = dict(enumerate(open("imagenet_1k_labels.txt")))


    # Load the model
    model_name = 'nest_base'
    model = timm.create_model('nest_base_jx.goog_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    img = Image.open('jmu_base.png')
    plt.imshow(np.array(img))
    plt.show()

    x = transforms(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)

    probs = logits.softmax(dim=1)
    top_probs, top_idcs = probs[0].topk(5)

    for i, (idx_, prob_) in enumerate(zip(top_idcs, top_probs)):
        idx = idx_.item()
        prob = prob_.item()
        clazz = imagenet_labels[idx].strip()
        print(f"{i}: {clazz:<45} --- {prob:.4f}")
