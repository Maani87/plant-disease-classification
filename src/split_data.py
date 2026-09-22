"""
Utility functions for splitting image classification datasets
into train, validation and test sets.
"""

import os
import shutil
import random
from pathlib import Path


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
):
    """
    Split image dataset into train, validation and test folders.

    Args:
        source_dir (str): Path to original dataset.
                         Each class should have its own folder.

        output_dir (str): Path where train/val/test folders will be created.

        train_ratio (float): Percentage of data used for training.

        val_ratio (float): Percentage of data used for validation.

        test_ratio (float): Percentage of data used for testing.

        seed (int): Random seed for reproducibility.
    """

    # Check ratios
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError(
            "Train, validation and test ratios must sum to 1."
        )

    random.seed(seed)

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    classes = [
        folder for folder in os.listdir(source_dir)
        if (source_dir / folder).is_dir()
    ]


    for class_name in classes:

        class_dir = source_dir / class_name

        images = list(class_dir.glob("*"))

        # Shuffle images
        random.shuffle(images)


        total_images = len(images)

        train_end = int(total_images * train_ratio)
        val_end = int(
            total_images * (train_ratio + val_ratio)
        )


        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]


        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }


        for split_name, split_images in splits.items():

            save_dir = (
                output_dir /
                split_name /
                class_name
            )

            save_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            for image_path in split_images:

                shutil.copy(
                    image_path,
                    save_dir / image_path.name
                )


        print(
            f"{class_name}: "
            f"{len(train_images)} train, "
            f"{len(val_images)} val, "
            f"{len(test_images)} test"
        )
