# python
from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.wrn import WideResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=6, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_dir', type=str, default='data/', help='dataset directory')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ece_score(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bins = torch.bucketize(probs.max(1).values, bin_boundaries, right=True) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bins == i
        if mask.any():
            acc = (probs[mask].argmax(1) == labels[mask]).float().mean()
            conf = probs[mask].max(1).values.mean()
            ece += (mask.float().mean() * torch.abs(acc - conf))
    return ece.item()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    transform = transforms.ToTensor()

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                transform=transform)
    idx = [i for i, t in enumerate(trainset.targets) if t < 5]
    trainset = Subset(trainset, idx)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    shifted = torch.load(data_dir / 'cifar10_shift.pt')
    testloader = DataLoader(torch.utils.data.TensorDataset(shifted['data'],
                                                           shifted['targets']),
                            batch_size=args.batch_size)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = WideResNet(num_classes=5)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        for x, y in tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = criterion(out, y.to(device))
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), data_dir / 'wrn_map.pt')

    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(testloader, desc="Evaluating"):
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(y)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    acc = (probs.argmax(1) == labels).float().mean().item()
    ece = ece_score(probs, labels)

    torch.save({'acc': acc, 'ece': ece}, data_dir / 'map_scores.pt')
    torch.save({'probs': probs, 'labels': labels}, data_dir / 'map_probs.pt')

    print(f"ECE_MAP={ece}")
    print(f"ACC_MAP={acc}")


if __name__ == '__main__':
    main()