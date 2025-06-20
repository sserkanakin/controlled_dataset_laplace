# Controlled Dataset Laplace

[![GitHub Repo](https://img.shields.io/badge/github-repo-blue)](https://github.com/your/repo)
[![DOI](https://img.shields.io/badge/zenodo-doi-green)](https://zenodo.org/badge/latestdoi/12345)

This repository demonstrates the effect of a constant colour shift on calibration of a
CIFAR-10 classifier and how a Laplace approximation can mitigate it.

## Installation

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
# create shifted dataset
python src/make_colour_shift.py

# train MAP estimate
python src/train_map.py

# apply Laplace approximation
python src/train_laplace.py
```

The notebook resides in the `notebooks/` directory.

## References
- Guo et al. 2017 *On Calibration of Modern Neural Networks*
- Ritter et al. 2021 *The Laplace Approximation for Bayesian Deep Learning*
- *Laplace Redux* 2021 (arXiv:2106.14806)
- Van Gemert 2024 *Research Guidelines in Deep Learning*
