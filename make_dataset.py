"""Create the Color-CIFAR10-Shift control dataset.

This script downloads CIFAR-10, saves the training images unchanged,
and generates a colour-shifted control test set. Images are saved as PNG
files in ``data/train`` and ``data/test_control``. Additionally, a grid of
example images is written to ``examples/grid.png``. Optionally, the dataset
can be uploaded to the HuggingFace Hub.

The script uses only ``torch``, ``torchvision``, ``datasets``, ``numpy``,
``pillow`` and ``matplotlib``.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from datasets import DatasetDict, load_dataset
import matplotlib.pyplot as plt
import torchvision


SHIFT_MAP: Dict[str, Tuple[int, int, int]] = {
    "airplane": (40, 0, 0),
    "automobile": (0, 40, 0),
    "bird": (0, 0, 40),
    "cat": (40, 40, 0),
    "deer": (40, 0, 40),
    "dog": (0, 40, 40),
    "frog": (60, 30, 0),
    "horse": (30, 60, 0),
    "ship": (0, 30, 60),
    "truck": (60, 0, 30),
}


def _apply_shift(img: Image.Image, shift: Tuple[int, int, int]) -> Image.Image:
    """Add an RGB shift to ``img`` and clamp to ``[0, 255]``."""
    arr = np.array(img, dtype=np.int16)
    arr[:, :, 0] += shift[0]
    arr[:, :, 1] += shift[1]
    arr[:, :, 2] += shift[2]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _save_split(dataset: torchvision.datasets.CIFAR10, out_dir: str, *, shift: bool) -> None:
    """Save a CIFAR10 split as PNG images.

    Args:
        dataset: ``torchvision`` dataset split.
        out_dir: directory to save images into.
        shift: if ``True`` apply class-specific colour shifts.
    """

    for idx, (img, label) in enumerate(dataset):
        cls = dataset.classes[label]
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        if shift:
            img = _apply_shift(img, SHIFT_MAP[cls])
        img.save(os.path.join(cls_dir, f"{idx:05d}.png"))


def create_dataset(data_root: str = "data") -> None:
    """Download CIFAR-10 and create the control dataset."""

    train = torchvision.datasets.CIFAR10(
        root="./cifar10", train=True, download=True
    )
    test = torchvision.datasets.CIFAR10(
        root="./cifar10", train=False, download=True
    )

    _save_split(train, os.path.join(data_root, "train"), shift=False)
    _save_split(test, os.path.join(data_root, "test_control"), shift=True)


def build_example_grid(data_dir: str = "data/test_control", out_path: str = "examples/grid.png") -> None:
    """Create a 2x5 grid of example images from the control test set."""

    classes = sorted(os.listdir(data_dir))
    images = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        img_file = sorted(os.listdir(cls_dir))[0]
        images.append(Image.open(os.path.join(cls_dir, img_file)))

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, img, cls in zip(axes.ravel(), images, classes):
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def export_to_hub(data_root: str = "data", repo_name: str = "color-cifar10-shift") -> None:
    """Export the dataset to the HuggingFace Hub."""

    train_ds = load_dataset("imagefolder", data_dir=os.path.join(data_root, "train"), split="train")
    test_ds = load_dataset(
        "imagefolder", data_dir=os.path.join(data_root, "test_control"), split="train"
    )
    dataset = DatasetDict({"train": train_ds, "test_control": test_ds})
    dataset.push_to_hub(repo_name)


if __name__ == "__main__":
    create_dataset()
    build_example_grid()
    # Uncomment the following line and login with ``huggingface-cli`` to upload
    # export_to_hub()
