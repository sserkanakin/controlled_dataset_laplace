# A Surgical Test for Calibration under Colour Shift

## 1 Motivation
Imagine a self-driving car misreading a traffic light because of tinted weather. Such colour shifts challenge model calibration [Guo et al., 2017]. Laplace approximations can help [Daxberger et al., 2021].

## 2 Hypothesis
"Last-layer K-FAC Laplace halves the Expected Calibration Error introduced by a fixed colour shift, without hurting accuracy."

## 3 Control dataset
The dataset applies deterministic RGB offsets per class. Example samples are shown below.

![Grid](examples/grid.png)

## 4 Baseline results
The table summarises the effect of colour shift and Laplace correction.

![ECE bars](examples/ece_bars.png)

## 5 Try it yourself
[Dataset on HF](https://huggingface.co/datasets/color-cifar10-shift) | [Colab](https://colab.research.google.com/github/<USER>/color-cifar10-shift/blob/main/notebooks/demo.ipynb) | [Repo](https://github.com/<USER>/color-cifar10-shift)
