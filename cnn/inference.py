import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from cnn.cnn_module import CNNLightning
from cnn.train import CIFAR_CFG

intermediate_outputs = []


def hook_fn(module, input, output):
    intermediate_outputs.append(output)


if __name__ == '__main__':
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    saved_image_path = 'imgs/jmu.png'

    if os.path.exists(saved_image_path):
        img = Image.open(saved_image_path)
        img_tensor = val_transform(img)
    else:
        val_set = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2, persistent_workers=True)

        dataiter = iter(val_loader)
        images, labels = next(dataiter)
        img_tensor = images[0]

        img = img_tensor.permute(1, 2, 0) * 0.5 + 0.5
        img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
        img.save(saved_image_path)

    plt.imshow(img_tensor.permute(1, 2, 0) * 0.5 + 0.5)
    plt.axis('off')
    plt.show()

    img_tensor = img_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'cerulean-sunset-11-epoch=87-val_acc=0.00.ckpt'

    model = CNNLightning.load_from_checkpoint(checkpoint_path, num_classes=CIFAR_CFG['num_classes'], input_size=CIFAR_CFG['input_size'])

    model = model.to(device)
    model.eval()

    hooks = []
    for i in range(1, 7):
        hook = getattr(model, f'conv{i}').register_forward_hook(hook_fn)
        hooks.append(hook)
        print(f"Registered Hook {i}")

    with torch.no_grad():
        output = model(img_tensor.to(device))
        _, predicted = torch.max(output, dim=1)

    print(f"Predicted class: {CLASSES[predicted.item()]}")

    for hook in hooks:
        hook.remove()

    images_per_row = 8
    max_images = 8

    for layer_num, layer_activation in enumerate(intermediate_outputs):
        # Skip layers with no feature map (like pooling and dropout layers)
        if len(layer_activation.size()) != 4:
            continue

        # This is the number of features in the feature map
        n_features = layer_activation.size(1)  # Number of channels
        n_features = min(n_features, max_images)

        # The feature map has shape (batch_size, n_features, height, width)
        size = layer_activation.size(2)

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, col * images_per_row + row].cpu().detach().numpy()

                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std() + 1e-5
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f'imgs/layer_{layer_num}_activations.png', bbox_inches='tight', pad_inches=0)

    plt.show()