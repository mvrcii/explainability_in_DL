import os
import random

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.datasets.cifar import CIFAR10

import wandb
from cnn.cnn_module import CNNLightning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        CLASSES = ['cat', 'dog']

        # Let's log 4 sample image predictions from first batch
        if batch_idx == 0:
            n = 4
            x, y = batch
            images = [img for img in x[:n]]
            captions = [(f'Ground Truth: {y_i}\n'
                         f'Prediction: {y_pred}\n'
                         f'Label: {CLASSES[y_i]}') for y_i, y_pred in zip(y[:n], outputs[:n])]

            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(key='sample_images', images=images, caption=captions)

            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table', columns=columns, data=data)


def get_transforms():
    img_size = (CONFIG['input_size'], CONFIG['input_size'])

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return train_transform, val_transform


def get_dataset(subset_size=None):
    train_transform, val_transform = get_transforms()

    if CONFIG['type'] == 'cat_dog':
        # Load cat and dog dataset
        train_set_full = datasets.ImageFolder(root='./data/cat_dog/training_set', transform=train_transform)
        val_set_full = datasets.ImageFolder(root='./data/cat_dog/test_set', transform=val_transform)

        # If a subset size is specified, randomly sample from the datasets
        if subset_size:
            train_indices = random.sample(range(len(train_set_full)), subset_size)
            val_indices = random.sample(range(len(val_set_full)), subset_size)
            train_set = Subset(train_set_full, train_indices)
            val_set = Subset(val_set_full, val_indices)
        else:
            train_set = train_set_full
            val_set = val_set_full

    elif CONFIG['type'] == 'cifar':
        # Load the CIFAR10 dataset
        train_set = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        val_set = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    else:
        raise ValueError('Invalid dataset type')

    return train_set, val_set


def get_data_loaders(batch_size=128, num_workers=8):
    train_set, val_set = get_dataset(subset_size=CONFIG['subset_size'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=True)
    return train_loader, val_loader


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CIFAR_CFG = {
    'type': 'cifar',
    'input_size': 32,
    'num_classes': 10,
    'batch_size': 128,
    'subset_size': None,  # Set to None to use the full dataset
    'labels': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}

CAT_DOG_CFG = {
    'type': 'cat_dog',
    'input_size': 128,
    'num_classes': 2,
    'batch_size': 4,
    'subset_size': 100,
    'labels': ['cat', 'dog']
}

if __name__ == '__main__':
    CONFIG = CAT_DOG_CFG

    classes = CONFIG['labels']
    train_loader, val_loader = get_data_loaders(batch_size=CONFIG['batch_size'])

    wandb_logger = WandbLogger(project="seminar")
    log_predictions_callback = LogPredictionsCallback()

    model = CNNLightning(
        input_size=CONFIG['input_size'],
        num_classes=CONFIG['num_classes']
    )
    model.to(get_device())

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        dirpath='checkpoints',
        filename=f'{{epoch:02d}}-{{val_accuracy:.2f}}'
    )

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=10,
        callbacks=[
            checkpoint_callback,
            log_predictions_callback
        ]
    )

    trainer.fit(model, train_loader, val_loader)
