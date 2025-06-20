"""Plot example grid and ECE bar chart."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


EXAMPLES_DIR = Path("examples")


def load_metrics(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_grid(grid_path: Path) -> None:
    if grid_path.exists():
        return
    raise FileNotFoundError(f"Example grid not found at {grid_path}")


def plot_ece(metrics: dict, out_path: Path) -> None:
    labels = ["Orig", "Shift", "Laplace"]
    eces = [metrics["ece_orig"], metrics["ece_shift"], metrics["ece_laplace"]]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    plt.figure(figsize=(4, 3))
    plt.bar(labels, eces, color=colors)
    plt.ylabel("ECE")
    plt.title("Expected Calibration Error")
    plt.tight_layout()
    plt.savefig(out_path)


def main() -> None:
    examples_dir = EXAMPLES_DIR
    grid_path = examples_dir / "grid.png"
    metrics_path = examples_dir / "metrics.json"
    ece_path = examples_dir / "ece_bars.png"

    plot_grid(grid_path)
    metrics = load_metrics(metrics_path)
    plot_ece(metrics, ece_path)


if __name__ == "__main__":
    main()
