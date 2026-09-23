# API reference

Exact contracts of the public objects: parameters, defaults, shapes, units,
conventions and restrictions.  Concepts are in {doc}`../explanation/index`,
complete workflows in the {doc}`examples <../examples/index>`.

| Page | Module | Contents |
| --- | --- | --- |
| {doc}`functions` | `bartorch` | Fourier and wavelet transforms, thresholding, array functions, runtime settings |
| {doc}`tools` | `bartorch.tools` | BART's commands as functions: simulation, sampling, calibration, reconstruction, registration and metrics |
| {doc}`linop` | `bartorch.linop` | Linear operators, operator algebra and MRI encoding operators |
| {doc}`nlop` | `bartorch.nlop` | Nonlinear operators, MRI and signal models, Gauss-Newton methods |
| {doc}`optim` | `bartorch.optim` | Iterative solvers, iteration blocks and data scaling |
| {doc}`priors` | `bartorch.priors` | Regularization terms, plug-and-play priors and denoisers |
| {doc}`apps` | `bartorch.apps` | BART reconstruction pipelines assembled from operators and solvers |
| {doc}`learning` | `bartorch.learning` | Adapters for neural networks and unrolled iterations |
| {doc}`interop` | `bartorch.interop` | DeepInverse physics adapter |
| {doc}`io` | `bartorch.io` | CFL files |
| {doc}`cli` | `bartorch.cli` | The `bartorch` command line |

```{toctree}
:hidden:

functions
tools
linop
nlop
optim
priors
apps
learning
interop
io
cli
```
