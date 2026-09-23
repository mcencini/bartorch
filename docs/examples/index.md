# Examples

Reconstructions executed when the documentation is built, so every figure and
printed number is produced by the code shown.  The examples use a BrainWeb
segmentation (`brainweb-dl`) as the phantom.  Concepts are in
{doc}`../explanation/index`, interfaces in {doc}`../api/index`.

| Section | Examples | Covers |
| --- | --- | --- |
| {doc}`Basics <../auto_examples/01-basics/index>` | 3 | Cartesian SENSE with ESPIRiT through a BART command; the same problem as an operator and a solver; noise prewhitening |
| {doc}`Non-Cartesian imaging <../auto_examples/02-non-cartesian/index>` | 2 | Trajectories, the non-uniform Fourier transform and its normal operator; radial SENSE |
| {doc}`Applications <../auto_examples/03-applications/index>` | 2 | Dynamic golden-angle radial MRI; subspace-constrained $T_1$ mapping |
| {doc}`Model-based reconstruction <../auto_examples/04-model-based/index>` | 2 | Nonlinear inversion; parameter maps estimated from k-space |
| {doc}`Deep learning <../auto_examples/05-deep-learning/index>` | 1 | An unrolled MoDL network on BART's ADMM |

The scripts need `pip install bartorch brainweb-dl matplotlib cmap`, and the
deep-learning example also `lightning torchio monai deepinv`.  Each page can
be downloaded as a script or a notebook.

```{toctree}
:hidden:

../auto_examples/01-basics/index
../auto_examples/02-non-cartesian/index
../auto_examples/03-applications/index
../auto_examples/04-model-based/index
../auto_examples/05-deep-learning/index
```
