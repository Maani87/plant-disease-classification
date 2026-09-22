"""
Predict Plant Disease from a single image.
"""

import argparse

import torch
from torch import nn

from PIL import Image

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0



# =====================
# Arguments
# =====================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="Path to image"
)

args = parser.parse_args()



# =====================
# Configuration
# =====================

MODEL_PATH = (
    "models/"
    "efficientnet_b0_plant_disease.pth"
)

TRAIN_DIR = "new/train"


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# =====================
# Classes
# =====================

dataset = ImageFolder(
    TRAIN_DIR
)

class_names = dataset.classes



# =====================
# Transform
# =====================

transform = transforms.Compose([

    transforms.Resize(
        (256,256)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])



# =====================
# Load Model
# =====================

model = efficientnet_b0(
    weights=None
)


model.classifier = nn.Sequential(

    nn.Dropout(0.2),

    nn.Linear(
        1280,
        len(class_names)
    )
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)


model.to(device)

model.eval()



# =====================
# Load Image
# =====================

image = Image.open(
    args.image
).convert("RGB")


image_tensor = transform(
    image
).unsqueeze(0)



# =====================
# Prediction
# =====================

with torch.inference_mode():

    output = model(
        image_tensor.to(device)
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )


    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )



print("\n===== Prediction =====")

print(
    f"Class: {class_names[prediction.item()]}"
)

print(
    f"Confidence: {confidence.item()*100:.2f}%"
)