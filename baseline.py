"""Evaluate WideResNet on Color-CIFAR10-Shift.

This script reproduces the baseline experiment described in the blog. It
loads a pretrained WideResNet-28-10, evaluates on the original CIFAR-10 test
set and on the colour-shifted control test set, and then applies Laplace Redux
for improved calibration. Accuracy and expected calibration error (ECE) are
reported for both datasets.
"""

from __future__ import annotations

import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from sklearn.metrics import accuracy_score
from laplace import Laplace


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Return Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    digitized = np.digitize(np.max(probs, axis=1), bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = digitized == i
        if mask.any():
            acc = (np.argmax(probs[mask], 1) == labels[mask]).mean()
            conf = np.max(probs[mask], axis=1).mean()
            ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return float(ece)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """Evaluate ``model`` and return accuracy and ECE."""
    is_laplace = isinstance(model, Laplace)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if is_laplace:
                probs = model(x, link_approx="probit").cpu()
            else:
                logits = model(x).cpu()
                probs = torch.softmax(logits, dim=1)
            preds.append(probs)
            labels.append(y)
    probs = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    ece = compute_ece(probs, labels)
    return float(acc), float(ece)


def load_model(num_classes: int = 10) -> torch.nn.Module:
    """Load a WideResNet-28-10 pretrained on CIFAR-10."""
    try:
        weights = torchvision.models.Wide_ResNet28_10_Weights.DEFAULT
        model = torchvision.models.wide_resnet28_10(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = torchvision.models.wide_resnet28_10(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def main(data_root: str = "data") -> None:
    """Run the baseline evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    orig_test = CIFAR10(root="./cifar10", train=False, download=True, transform=transform)
    shift_test = ImageFolder(os.path.join(data_root, "test_control"), transform=transform)

    loader = DataLoader(orig_test, batch_size=128, shuffle=False, num_workers=2)
    shift_loader = DataLoader(shift_test, batch_size=128, shuffle=False, num_workers=2)

    model = load_model()
    model.to(device)

    acc_orig, ece_orig = evaluate(model, loader, device)
    acc_shift, ece_shift = evaluate(model, shift_loader, device)

    la = Laplace(model, "classification", subset_of_weights="last_layer", hessian_structure="kron")
    la.fit(loader)
    la.optimize_prior_precision(method="marglik")

    la_acc_orig, la_ece_orig = evaluate(la, loader, device)
    la_acc_shift, la_ece_shift = evaluate(la, shift_loader, device)

    print("| model | accuracy original | ECE original | accuracy color-shift | ECE color-shift |")
    print("|-------|-------------------|--------------|----------------------|-----------------|")
    print(f"| MAP | {acc_orig*100:.1f}% | {ece_orig:.3f} | {acc_shift*100:.1f}% | {ece_shift:.3f} |")
    print(f"| Laplace | {la_acc_orig*100:.1f}% | {la_ece_orig:.3f} | {la_acc_shift*100:.1f}% | {la_ece_shift:.3f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument("--data-root", default="data", help="dataset directory")
    args = parser.parse_args()
    main(args.data_root)
