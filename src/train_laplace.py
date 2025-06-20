# python
"""Wrap trained model with Laplace approximation and re-evaluate."""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from laplace import Laplace

from models.wrn import WideResNet
from train_map import ece_score, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hessian', type=str, default='kron', help='hessian structure')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_dir', type=str, default='data/', help='dataset directory')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    transform = transforms.ToTensor()

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                transform=transform)
    idx = [i for i, t in enumerate(trainset.targets) if t < 5]
    trainset = Subset(trainset, idx)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=False)

    shifted = torch.load(data_dir / 'cifar10_shift.pt')
    testloader = DataLoader(torch.utils.data.TensorDataset(shifted['data'],
                                                           shifted['targets']),
                            batch_size=128)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = WideResNet(num_classes=5)
    model.load_state_dict(torch.load(data_dir / 'wrn_map.pt', map_location=device))
    model.to(device)
    model.eval()

    la = Laplace(model, likelihood='classification', subset_of_weights='last_layer',
                 hessian_structure=args.hessian)
    la.fit(trainloader)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(testloader, desc="Evaluating"):
            f_preds = la(x.to(device))
            probs = torch.softmax(f_preds, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(y)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    acc = (probs.argmax(1) == labels).float().mean().item()
    ece = ece_score(probs, labels)

    torch.save({'probs': probs, 'labels': labels}, data_dir / 'laplace_probs.pt')

    map_scores = torch.load(data_dir / 'map_scores.pt')
    ece_map = map_scores['ece']
    acc_map = map_scores['acc']

    print(f"ECE_MAP={ece_map}")
    print(f"ECE_LAPLACE={ece}")
    print(f"ACC_MAP={acc_map}")
    print(f"ACC_LAPLACE={acc}")

    if ece >= 0.5 * ece_map or abs(acc_map - acc) > 0.002:
        exit(1)


if __name__ == '__main__':
    main()