# Color-CIFAR10-Shift

This repository provides code to create and evaluate the **Color-CIFAR10-Shift** control dataset.

## Setup

Install dependencies:

```bash
pip install torch torchvision datasets laplace-torch numpy pillow matplotlib scikit-learn
```

## Dataset generation

Run the following to download CIFAR-10 and create the colour-shifted control test set:

```bash
python make_dataset.py
```

Images are stored under `data/train` and `data/test_control`. An example grid is written to `examples/grid.png`.

## Baseline experiment

After generating the dataset you can reproduce the results reported in the blog with:

```bash
python baseline.py
```

The script prints a table with accuracy and Expected Calibration Error on the original and shifted test sets before and after applying Laplace Redux.

## Further reading

See `blog.md` for a short post discussing the motivation, dataset and results.
