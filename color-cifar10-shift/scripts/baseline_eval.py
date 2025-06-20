"""Baseline evaluation with and without Laplace."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import wide_resnet50_2
from torch.utils.data import DataLoader
from laplace import Laplace
from sklearn.metrics import accuracy_score

def load_control_dataset() -> torch.utils.data.Dataset:
    imgs = []
    labels = []
    control_root = Path("data/test_control")
    for label_dir in sorted(control_root.iterdir()):
        for img_path in label_dir.glob("*.png"):
            img = T.ToTensor()(PILImage.open(img_path))
            imgs.append(img)
            labels.append(int(label_dir.name))
    tensor_imgs = torch.stack(imgs)
    tensor_labels = torch.tensor(labels)
    return torch.utils.data.TensorDataset(tensor_imgs, tensor_labels)


def evaluate(model, loader):
    logits = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device)
            out = model(x)
            logits.append(out.cpu())
            targets.append(y)
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    preds = logits.argmax(1)
    acc = accuracy_score(targets.numpy(), preds.numpy())
    probs = F.softmax(logits, dim=1)
    conf = probs.max(1).values
    correctness = preds.eq(targets)
    bins = torch.linspace(0, 1, 11)
    bin_ids = torch.bucketize(conf, bins) - 1
    ece = 0.0
    for i in range(len(bins) - 1):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        ece += torch.abs(conf[mask].mean() - correctness[mask].float().mean()) * mask.sum() / len(conf)
    return acc, ece.item()


def main() -> None:
    torch.manual_seed(0)
    test_ds = CIFAR10(root="tmp", train=False, download=True, transform=T.ToTensor())
    control_ds = load_control_dataset()
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    control_loader = DataLoader(control_ds, batch_size=128, shuffle=False)

    model = wide_resnet50_2(pretrained=True).eval()

    acc_orig, ece_orig = evaluate(model, test_loader)
    acc_shift, ece_shift = evaluate(model, control_loader)

    la = Laplace(model, 'classification', subset_of_weights='last_layer', hessian_structure='kron')
    la.fit(test_loader)
    la.optimize_prior_precision()

    acc_laplace, ece_laplace = evaluate(la, control_loader)

    metrics = {
        'acc_orig': acc_orig,
        'ece_orig': ece_orig,
        'acc_shift': acc_shift,
        'ece_shift': ece_shift,
        'acc_laplace': acc_laplace,
        'ece_laplace': ece_laplace,
    }

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    with open(examples_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    table = (
        "| Dataset | Acc | ECE |\n"
        "|---------|-----|-----|\n"
        f"| Original | {acc_orig:.3f} | {ece_orig:.3f} |\n"
        f"| Shifted | {acc_shift:.3f} | {ece_shift:.3f} |\n"
        f"| Laplace | {acc_laplace:.3f} | {ece_laplace:.3f} |\n"
    )
    print(table)


if __name__ == "__main__":
    main()
