"""
Evaluation utilities for image classification models.

Includes:
- Test evaluation
- Classification report
- Confusion matrix
"""


import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
from pathlib import Path

from torchvision import transforms
from torchvision.models import efficientnet_b0

import data_setup

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)



def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    class_names,
    loss_fn,
    device,
    save_path="results/confusion_matrix.png"
):
    """
    Evaluate model on test dataset.
    """


    model.eval()

    test_loss = 0

    y_true = []
    y_pred = []


    with torch.inference_mode():

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)


            outputs = model(X)


            loss = loss_fn(
                outputs,
                y
            )

            test_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            y_true.extend(
                y.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )



    # Average loss

    test_loss /= len(dataloader)



    # Accuracy

    test_accuracy = accuracy_score(
        y_true,
        y_pred
    )



    print("\n===== Test Results =====")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )



    print(
        "\n===== Classification Report ====="
    )


    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )


    print(report)



    # Save classification report

    with open(
        "results/classification_report.txt",
        "w"
    ) as f:

        f.write(report)



    # Confusion Matrix

    cm = confusion_matrix(
        y_true,
        y_pred
    )



    plt.figure(
        figsize=(16,14)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )


    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )


    plt.xticks(
        rotation=90
    )


    plt.tight_layout()



    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()



    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }





def main():

    # Device

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Using device: {device}"
    )



    # Paths

    train_dir = "new/train"
    val_dir = "new/val"
    test_dir = "new/test"


    model_path = (
        "models/"
        "efficientnet_b0_plant_disease.pth"
    )



    # Transform

    test_transform = transforms.Compose([

        transforms.Resize(
            (256,256)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )

    ])




    # DataLoader

    _, _, test_dataloader, class_names = data_setup.create_dataloaders(

        train_dir=train_dir,

        val_dir=val_dir,

        test_dir=test_dir,

        train_transform=test_transform,

        test_transform=test_transform,

        batch_size=32
    )



    # Model

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



    # Load weights

    model.load_state_dict(

        torch.load(

            model_path,

            map_location=device,

            weights_only=True

        )

    )



    model.to(device)



    # Results folder

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )



    # Evaluate

    evaluate_model(

        model=model,

        dataloader=test_dataloader,

        class_names=class_names,

        loss_fn=nn.CrossEntropyLoss(),

        device=device,

        save_path=
        "results/confusion_matrix.png"

    )





if __name__ == "__main__":

    main()