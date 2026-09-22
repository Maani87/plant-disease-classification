"""
Train Plant Disease Classification model using EfficientNet-B0.

Pipeline:
- Data loading
- Training
- Saving best model
- Test evaluation
- Confusion Matrix
- Grad-CAM visualization
"""

import os
import torch
from torch import nn
import data_setup, engine, utils, visualization

from torchvision import transforms
import argparse
from pathlib import Path
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--patience", type=int, default=10)

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
PATIENCE = args.patience



# Setup directories
train_dir = "new/train"
val_dir = "new/val"
test_dir = "new/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])



# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    test_dir=test_dir,
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=BATCH_SIZE
)

weights = EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
model = efficientnet_b0(weights=weights).to(device)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.2, inplace=True),
    nn.Linear(1280, len(class_names))
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=1e-4)

# Start training with help from engine.py
history = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    patience=PATIENCE,
    device=device
)

visualization.save_training_curves(
    history
)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="efficientnet_b0_plant_disease.pth")
