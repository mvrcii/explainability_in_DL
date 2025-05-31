import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.adamw import AdamW


class CNNLightning(LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv5 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv6 = nn.Conv2d(128, 128, 3, padding='same')

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        # Dynamically calculate the flattened size
        reduced_size = input_size // (2 ** 3)  # Assuming 3 pooling layers
        flattened_size = 128 * reduced_size * reduced_size

        # Fully connected layer
        self.fc = nn.Linear(flattened_size, num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout2(x)

        # Third block
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.dropout3(x)

        # Flatten and pass through the dense layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        inputs = inputs.cpu()
        plt.imshow(inputs[0].permute(1, 2, 0) * 0.5 + 0.5)
        plt.axes().set_title(f'Ground Truth: {labels[0]}\nPrediction: {torch.argmax(logits[0])}')

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_accuracy', accuracy, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)

        loss = self.loss(logits, labels)

        inputs = inputs.cpu()
        plt.imshow(inputs[0].permute(1, 2, 0) * 0.5 + 0.5)
        plt.axes().set_title(f'Ground Truth: {labels[0]}\nPrediction: {torch.argmax(logits[0])}')

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_accuracy', accuracy, prog_bar=True, on_epoch=True, on_step=False)

        return preds

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-4, weight_decay=0.0001)
