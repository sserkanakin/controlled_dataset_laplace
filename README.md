# Controlled Dataset Laplace

[![GitHub Repo](https://img.shields.io/badge/github-repo-blue)](https://github.com/your/repo)
[![DOI](https://img.shields.io/badge/zenodo-doi-green)](https://zenodo.org/badge/latestdoi/12345)

This repository demonstrates how a uniform RGB bias affects calibration of a
CIFAR-10 model and how a Laplace approximation can recover the lost reliability.
We keep only the first five classes for speed. When the (+32,+32,+32) shift is applied, our run saw ECE rise to 0.103 with MAP accuracy 67.7%. Applying the Laplace approximation kept accuracy roughly equal but raised the ECE to 0.388.

## Installation

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
# create shifted dataset
python src/make_colour_shift.py --seed 0

# train MAP estimate
python src/train_map.py --epochs 6

# apply Laplace approximation
python src/train_laplace.py --hessian kron
```

The second script prints `ECE_MAP` and `ACC_MAP`. The third script prints all
four lines and exits with code `1` if calibration does not improve sufficiently.

Open `notebooks/inspect.ipynb` to plot reliability diagrams and scatter plots
from the saved predictions.

## References
- Guo et al. 2017 *On Calibration of Modern Neural Networks*
- Ritter et al. 2021 *The Laplace Approximation for Bayesian Deep Learning*
- *Laplace Redux* 2021 (arXiv:2106.14806)
- Van Gemert 2024 *Research Guidelines in Deep Learning*
