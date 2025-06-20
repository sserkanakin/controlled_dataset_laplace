"""Generate and optionally upload Color-CIFAR10-Shift dataset."""

# Color-CIFAR10-Shift generator
# References: Guo 2017; Daxberger 2021
# Hypothesis: Last-layer K-FAC Laplace halves the Expected Calibration Error introduced by a fixed colour shift, without hurting accuracy.

import argparse
import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10


OFFSETS = {
    0: (20, 0, 0),
    1: (0, 20, 0),
    2: (0, 0, 20),
    3: (-20, 0, 0),
    4: (0, -20, 0),
    5: (0, 0, -20),
    6: (15, 15, 0),
    7: (0, 15, 15),
    8: (15, 0, 15),
    9: (-15, -15, 0),
}


def apply_offset(img: PILImage, offset: tuple[int, int, int]) -> PILImage:
    """Add an RGB offset to an image.

    Args:
        img: Input image.
        offset: RGB tuple to add.

    Returns:
        Offset image.
    """
    arr = np.array(img, dtype=np.int16)
    arr = np.clip(arr + np.array(offset, dtype=np.int16), 0, 255)
    return PILImage.fromarray(arr.astype(np.uint8))


def save_split(dataset: CIFAR10, root: Path, transform=None) -> list[dict]:
    """Save CIFAR-10 split to disk and return list for HF dataset."""
    data_list = []
    for idx, (img, label) in enumerate(tqdm(dataset, desc=f"Saving to {root}")):
        if transform:
            img = transform(img, OFFSETS[label])
        class_dir = root / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        file_path = class_dir / f"{idx:05d}.png"
        img.save(file_path)
        data_list.append({"image": str(file_path), "label": label})
    return data_list


def make_example_grid(root: Path, out_path: Path) -> None:
    """Create a 2x5 grid of examples from each class."""
    imgs = []
    for label in range(10):
        img_path = next((root / str(label)).glob("*.png"))
        imgs.append(PILImage.open(img_path))
    rows = []
    for i in range(2):
        row = np.hstack([np.array(imgs[j + 5 * i]) for j in range(5)])
        rows.append(row)
    grid = np.vstack(rows)
    PILImage.fromarray(grid).save(out_path)


def main(push: bool = False) -> None:
    """Generate dataset and optionally push to HuggingFace Hub."""
    data_dir = Path("data")
    train_root = data_dir / "train"
    test_control_root = data_dir / "test_control"
    examples_dir = Path("examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CIFAR10(root="tmp", train=True, download=True)
    test_ds = CIFAR10(root="tmp", train=False, download=True)

    train_data = save_split(train_ds, train_root)
    test_data = save_split(test_ds, test_control_root, transform=apply_offset)

    grid_path = examples_dir / "grid.png"
    make_example_grid(test_control_root, grid_path)

    if push:
        features = Features({"image": Image(), "label": Value("int64")})
        ds_dict = DatasetDict({
            "train": Dataset.from_list(train_data, features=features),
            "test_control": Dataset.from_list(test_data, features=features),
        })
        ds_dict.push_to_hub("color-cifar10-shift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push dataset to HF")
    args = parser.parse_args()
    main(push=args.push)
