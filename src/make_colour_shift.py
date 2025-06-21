"""Generate a colour shifted CIFAR-10 test set."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shift', type=int, default=32, help='RGB bias to add')
    parser.add_argument('--classes', type=int, default=5,
                        help='number of classes to keep')
    parser.add_argument('--out_dir', type=str, default='data/', help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root=out_dir, train=False, download=True,
                               transform=transform)

    data = []
    targets = []
    for img, label in testset:
        if label < args.classes:
            img = img + args.shift / 255.0
            img = torch.clamp(img, 0.0, 1.0)
            data.append(img)
            targets.append(label)

    tensor = torch.stack(data)
    labels = torch.tensor(targets)
    torch.save({'data': tensor, 'targets': labels}, out_dir / 'cifar10_shift.pt')


if __name__ == '__main__':
    main()
