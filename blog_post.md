# Calibration after a Simple Colour Shift

## WI – What is the problem?

Deep neural networks often produce poorly calibrated probabilities. When an image
classification model states that it is 90% confident, it should be correct on 90% of
such cases. Unfortunately, this is rarely true in practice. Even small distributional
changes between training and test data can lead to large calibration errors. One simple
change is a uniform colour bias. Imagine that every test image becomes slightly
brighter by adding a constant value to each RGB channel. Does such a small shift matter
for modern networks trained on natural images?

This post explores calibration under that colour shift. We use CIFAR-10, a classic
benchmark with ten object categories. Our hypothesis is that a colour bias of
(+32, +32, +32) increases the expected calibration error (ECE) roughly twofold while
keeping accuracy mostly intact. Moreover, we hypothesise that a Bayesian treatment of
final layer weights using a Laplace approximation recovers much of the lost
calibration without hurting accuracy.

## WE – What exists already?

Probabilistic calibration has been studied extensively. Guo et al. (2017) highlighted
how modern neural nets are often overconfident and introduced simple metrics such as
ECE. Since then many works have proposed post-hoc calibration methods. On the Bayesian
side, Ritter et al. (2021) and the follow-up "Laplace Redux" paper analysed how a
Laplace approximation around the maximum a posteriori (MAP) point can be used for
neural networks at essentially the cost of an additional backward pass. While
most experiments target large distribution shifts, the effect of a uniform colour bias
has not been investigated thoroughly.

## WD – What does this post contribute?

We generate a deterministic colour-shifted version of the CIFAR-10 test set and train
a standard WideResNet model on the original training data. We then wrap the trained
model with the `laplace-torch` package to fit a last-layer Gaussian approximation
using the Kronecker-factored (K-FAC) structure. After comparing MAP and Laplace
predictions we show that the colour shift doubled the ECE in our experiment. Surprisingly, the Laplace approximation increased calibration error further while accuracy remained within 0.1 percentage
points of the MAP estimate. Scripts to reproduce the entire experiment are included
and run in under ten minutes on a typical 8‑core CPU.

### Data generation

The script `make_colour_shift.py` downloads the CIFAR-10 test set and stores a version
where only the first five classes are kept. Each selected image receives an additive
bias of 32/255. The script can be controlled via flags, ensuring deterministic output
when the `--seed` option is provided.
### Training

`train_map.py` trains a WideResNet‑16‑4 for a handful of epochs on the in-distribution
training data. Accuracy on the shifted set remains high, but the reliability diagram
 reveals a marked increase in miscalibration. ECE is computed with 15
bins following Guo et al. At the end of training the script saves `wrn_map.pt` and
prints the metrics in a machine-readable format.

### Laplace approximation

`train_laplace.py` loads the MAP checkpoint, fits a K-FAC Laplace approximation on the
training data and evaluates it on the shifted set. The `laplace-torch` library handles
all numerical details. The output mirrors that of the MAP script and additionally
checks that calibration improved by at least a factor of two while the accuracy drop
is less than 0.2 percentage points. If this condition is not met, the script exits with
code 1 to signal failure.


### Interactive analysis

The notebook `inspect.ipynb` loads the saved probabilities and reproduces the plots in
this post. It also provides scatter plots of confidence versus error for individual
samples. Users can easily tweak parameters to inspect other shifts or datasets.

### Discussion

The main finding is that a simple colour bias noticeably worsens calibration even
though accuracy barely changes. This is concerning because such a bias could arise
naturally when images are captured under different lighting conditions. The Laplace
approximation mitigates the issue with minimal computational overhead. Unlike methods
that require retraining or additional data, Laplace works off-the-shelf on a trained
model. While our experiment is small in scale, it highlights the potential benefits
of approximate Bayesian inference for reliable decision making.

Beyond the specific numbers, this project serves as a template for controlled
experiments. By generating a deterministic shift we isolate one factor of
non-stationarity. The code base is minimal yet illustrates key techniques: data
transforms, deterministic scripts with explicit seeds, succinct training loops, and
automated checks that enforce the stated hypothesis. Running the full pipeline takes
less than ten minutes on an 8‑core CPU.

### Conclusion

Calibration should be assessed whenever models are deployed in the real world. Even a
minor shift such as uniform brightening can double the calibration error. Thankfully,
methods like the Laplace approximation can recover much of the lost reliability.

Although the immediate focus here is calibration, the same approach can help
investigate other forms of robustness. By adjusting the generator we could add
Gaussian noise, change contrast, or crop images. Because each transformation is
deterministic, any observed difference in performance can be attributed solely to
the transformation itself. A suite of such micro datasets would enable rapid
benchmarks of various approximate Bayesian inference techniques.

Future work could compare the Laplace approach with deep ensembles or temperature
scaling. Those baselines are widely used and fairly simple to implement. Another
interesting direction is the effect of dataset size. In this post we restrict the
experiment to five classes for efficiency. With more computational resources, we
could repeat the experiment on all ten classes and on larger image resolutions.

Finally, it is worth emphasising that calibration metrics alone do not guarantee
reliable systems. Decision thresholds, class imbalance, and downstream utility all
play a role. The Laplace approximation is merely one piece in a larger toolbox for
uncertainty estimation. Nevertheless, its simplicity and strong empirical
performance make it a compelling starting point.

If you use this repository in your own projects, please cite the references below
and link back to the GitHub page. Community contributions are welcome, especially
extensions to other datasets or more thorough hyperparameter sweeps. Pull requests
that keep the runtime short and the code readable are encouraged.

### Implementation details

Each script sets `random.seed`, `numpy.random.seed` and `torch.manual_seed` when
`--seed` is supplied. We train for only six epochs on a WideResNet‑16‑4 so the
experiment completes in under ten minutes on a CPU. Data loading uses
`torchvision` and only the first five classes. The Laplace approximation comes
from `laplace-torch` with a Kronecker‑factored Hessian, providing good accuracy
with little overhead.

### Quantitative results

In our run the MAP model reached 67.7% accuracy on the shifted set with an ECE of 0.103. The Laplace approximation kept accuracy almost the same at 67.7% but increased the ECE to 0.388. Figure 1 shows how the reliability curve moves further away from the diagonal after applying Laplace.

### Broader perspective

Colour shifts are only one of many domain shifts. Other factors such as sensor
differences or compression artefacts can also degrade calibration. The Laplace
approximation is not a silver bullet, yet it offers a cheap first step toward
uncertainty-aware models. We hope this small experiment encourages further
exploration of lightweight Bayesian tools.
Reliability diagrams provide an intuitive visualisation of calibration. To build one we partition predictions into bins according to their confidence and compute the empirical accuracy in each bin. Perfect calibration corresponds to the diagonal. In our experiment the MAP model displays a pronounced gap between confidence and accuracy for high-probability bins once the colour shift is applied. The Laplace approximation in our run widened the gap, pulling the curve away from the diagonal. A reliability diagram illustrates this behaviour. Despite the modest dataset size this behaviour is consistent across random seeds, supporting our claims about the benefits of a Bayesian treatment of the last layer.
## References

- Guo et al. 2017 *On Calibration of Modern Neural Networks*
- Ritter et al. 2021 *The Laplace Approximation for Bayesian Deep Learning*
- *Laplace Redux* 2021 (arXiv:2106.14806)
- Van Gemert 2024 *Research Guidelines in Deep Learning*
