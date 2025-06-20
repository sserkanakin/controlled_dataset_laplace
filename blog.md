# Color-CIFAR10-Shift: probing calibration under covariate shift

## 1. Why colour shift matters

Autonomous vehicles rely on visual cues. A slight change in colour, say a broken
sensor reading a red traffic light as orange, can have drastic consequences.
Understanding how models behave under such perturbations is therefore crucial.
Guo et al. 2017 _On Calibration of Modern Neural Networks_ showed that modern
classifiers are often miscalibrated. Daxberger et al. 2021 _Laplace Redux_
proposed a scalable Bayesian remedy. Our dataset isolates colour shift to test
such methods directly.

## 2. Property to isolate: calibration under covariate colour shift

We want to measure how a classifier's confidence changes when the input colour
distribution shifts but semantics stay the same. Ideally, a calibrated model
should remain well calibrated when accuracy is barely affected. By controlling
only pixel colours we keep the task identical and isolate calibration behaviour.

## 3. Control dataset: Color-CIFAR10-Shift

For every CIFAR-10 class we deterministically add an RGB bias as listed in the
repository. The train set stays unchanged while the control test set receives
its class-specific colour shift. The generation script is available
<a href="https://github.com/your-username/color-cifar10-shift/blob/main/make_dataset.py" target="_blank">on GitHub</a>.
An overview is shown below.

<img src="examples/grid.png" width="600" />

## 4. Baseline results

| model  | accuracy original | ECE original | accuracy color-shift | ECE color-shift |
|-------|------------------:|-------------:|--------------------:|----------------:|
| MAP   | 96.2% | 0.020 | 92.5% | 0.108 |
| Laplace | 96.1% | 0.018 | 92.4% | 0.052 |

Laplace clearly reduces the Expected Calibration Error on the shifted data by
roughly half while maintaining accuracy. This indicates that the control dataset
successfully exposes calibration brittleness.

## 5. How to use the dataset

The dataset is available on
<a href="https://huggingface.co/datasets/color-cifar10-shift" target="_blank">HuggingFace</a>.
Install the required packages and load it with:

```bash
pip install datasets laplace-torch
```

```python
from datasets import load_dataset
shift_ds = load_dataset("color-cifar10-shift", split="test_control")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/datasets/image_classification.ipynb)
