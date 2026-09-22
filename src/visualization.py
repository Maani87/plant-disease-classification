"""
Visualization utilities for training results.

Includes:
- Training loss curve
- Validation loss curve
- Training accuracy curve
- Validation accuracy curve
"""


import matplotlib.pyplot as plt
from pathlib import Path



def save_training_curves(history):
    """
    Save training and validation curves.

    Args:
        history: dictionary containing:
            train_loss
            train_acc
            val_loss
            val_acc
    """


    # Create results folder

    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )


    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )


    plt.figure(
        figsize=(12,5)
    )



    # Loss plot

    plt.subplot(
        1,
        2,
        1
    )


    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )


    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )


    plt.title(
        "Loss Curve"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.legend()



    # Accuracy plot

    plt.subplot(
        1,
        2,
        2
    )


    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy"
    )


    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )


    plt.title(
        "Accuracy Curve"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.legend()



    plt.tight_layout()



    plt.savefig(
        "results/training_curves.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



    print(
        "[INFO] Training curves saved to: results/training_curves.png"
    )
