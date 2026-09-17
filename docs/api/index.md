# API reference

| Page | Contents |
| --- | --- |
| {doc}`functions` | `bartorch`: Fourier and wavelet transforms, thresholding, array utilities, interpolation, settings |
| {doc}`linop` | `bartorch.linop`: the linear operator class, composition, elementary and MRI encoding operators |
| {doc}`priors` | `bartorch.priors`: regularization terms and denoisers |
| {doc}`optim` | `bartorch.optim`: iterative algorithms, and the data normalization a reconstruction applies |
| {doc}`nlop` | `bartorch.nlop`: the nonlinear operator class, composition, operators defined in Python or torch |
| {doc}`learning` | `bartorch.learning`: adapters between neural networks and this package's images and iterations |
| {doc}`apps` | `bartorch.apps`: BART's reconstruction pipelines, assembled from this package |
| {doc}`tools` | `bartorch.tools`: simulation, sampling and trajectories, coil calibration, reconstruction, and the operations around one -- resampling, registration, measurement |
| {doc}`io` | `bartorch.io`: CFL files |
| {doc}`interop` | `bartorch.interop`: an operator as a deepinv physics, for its samplers and its physics-based losses |

The concepts these objects implement are introduced in
{doc}`../explanation/index`, and complete workflows built from them are in the
{doc}`examples <../auto_examples/index>`.

```{toctree}
:hidden:

functions
linop
priors
optim
nlop
learning
apps
tools
io
interop
```
