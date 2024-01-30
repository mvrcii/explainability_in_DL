import json
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import timm
from PIL import Image
from openai import OpenAI
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
    imagenet_labels = dict(enumerate(open("imagenet_1k_labels.txt")))

    k = 5

    # Load the model
    model_name = 'nest_base'
    model = timm.create_model('nest_base_jx.goog_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Load and transform an image
    # response = requests.get('https://picsum.photos/id/237/200/300')
    # img = Image.open(BytesIO(response.content)).convert('RGB')

    # DALL-E 2 Image
    prompt = "Julius Maximilian in front of Würzburg's famous residence building with some green garden"
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="256x256",
        quality="standard",
        n=1,
    )
    response = requests.get(response.data[0].url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    plt.imshow(np.array(img))
    plt.show()

    img_input = transforms(img).unsqueeze(0)  # unsqueeze single image into batch of 1

    logits = model(img_input)
    probs = logits.softmax(dim=1)

    top_probs, top_idcs = probs[0].topk(k)

    for i, (idx_, prob_) in enumerate(zip(top_idcs, top_probs)):
        idx = idx_.item()
        prob = prob_.item()
        clazz = imagenet_labels[idx].strip()
        print(f"{i}: {clazz:<45} --- {prob:.4f}")
