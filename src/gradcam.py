"""
Grad-CAM visualization for PyTorch models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
import argparse

from torch import nn

from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.datasets import ImageFolder

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



def predict_image(
    model,
    image_path,
    class_names,
    device
):

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])


    image = Image.open(
        image_path
    ).convert("RGB")


    input_tensor = transform(
        image
    ).unsqueeze(0)


    with torch.inference_mode():

        output = model(
            input_tensor.to(device)
        )


        probability = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probability,
            dim=1
        )


    return (
        input_tensor,
        class_names[prediction.item()],
        confidence.item(),
        prediction.item()
    )



def generate_gradcam(
    model,
    image_tensor,
    target_layer,
    device,
    target_category
):

    model.eval()


    cam = GradCAM(
        model=model,
        target_layers=[
            target_layer
        ]
    )


    targets = [
        ClassifierOutputTarget(
            target_category
        )
    ]


    grayscale_cam = cam(
        input_tensor=image_tensor.to(device),
        targets=targets
    )


    return grayscale_cam[0]



def show_gradcam(
    image_path,
    cam,
    prediction
):

    image = Image.open(
        image_path
    ).convert("RGB")


    image = image.resize(
        (256,256)
    )


    image = np.array(
        image
    ) / 255.0



    visualization = show_cam_on_image(
        image,
        cam,
        use_rgb=True
    )


    plt.figure(
        figsize=(10,5)
    )


    plt.subplot(1,2,1)

    plt.imshow(
        image
    )

    plt.title(
        "Original Image"
    )

    plt.axis(
        "off"
    )


    plt.subplot(1,2,2)

    plt.imshow(
        visualization
    )

    plt.title(
        f"Grad-CAM\n{prediction}"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    # Create results folder

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )


    save_path = (
        f"results/"
        f"gradcam_{prediction}.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        f"Grad-CAM saved to: {save_path}"
    )


    plt.show()



def main():

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    MODEL_PATH = (
        "models/"
        "efficientnet_b0_plant_disease.pth"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image"
    )

    args = parser.parse_args()
    
    IMAGE_PATH = Path(args.image)



    # Load classes

    dataset = ImageFolder(
        "new/train"
    )


    class_names = dataset.classes



    # Load model

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
            map_location=DEVICE,
            weights_only=True
        )
    )


    model.to(DEVICE)



    # Prediction

    (
        image_tensor,
        prediction,
        confidence,
        prediction_index
    ) = predict_image(
        model=model,
        image_path=IMAGE_PATH,
        class_names=class_names,
        device=DEVICE
    )


    print("===== Prediction =====")

    print(
        f"Class: {prediction}"
    )

    print(
        f"Confidence: {confidence*100:.2f}%"
    )



    # Grad-CAM

    cam = generate_gradcam(
        model=model,
        image_tensor=image_tensor,
        target_layer=model.features[-1],
        device=DEVICE,
        target_category=prediction_index
    )


    # Show + Save

    show_gradcam(
        image_path=IMAGE_PATH,
        cam=cam,
        prediction=prediction
    )



if __name__ == "__main__":
    main()