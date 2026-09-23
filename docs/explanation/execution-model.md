# Interfaces and execution

bartorch exposes BART at two levels.  The command-style interface calls BART's
applications — `pics`, `ecalib`, `nlinv` — as functions of tensors.  The
composable interface exposes the objects those applications are built from:
the encoding operator, the regularization terms and the iterative algorithm,
which a reconstruction assembles in Python.  Both call the same embedded BART
through one C interface.

```{image} ../_static/architecture.svg
:class: only-light
:alt: PyTorch tensors enter the command-style and composable interfaces, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform Fourier transforms and linear algebra are served by substituted backends
:width: 100%
```

```{image} ../_static/architecture-dark.svg
:class: only-dark
:alt: PyTorch tensors enter the command-style and composable interfaces, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform Fourier transforms and linear algebra are served by substituted backends
:width: 100%
```

## The interfaces

| Interface | Unit | Autograd | Runs as |
| --- | --- | --- | --- |
| {mod}`bartorch.tools` | One BART command | No | The command, in this process |
| `bartorch.fft`, `fwt`, `rss`, ... ({doc}`../api/functions`) | One array operation | No | A BART command |
| `bartorch` command line ({mod}`bartorch.cli`) | A `bart` command line on CFL files | No | An app where one exists, otherwise the command |
| {mod}`bartorch.linop`, {mod}`bartorch.nlop` | An operator and its adjoint or derivative | Yes | A BART operator, or Python callbacks |
| {mod}`bartorch.optim`, {mod}`bartorch.priors` | A solver or one iteration step, and its terms | Yes, per solver | BART's iteration: CG in the library, proximal steps in Python over BART's operators |
| {mod}`bartorch.apps` | A BART application re-expressed with operators and solvers | As its solver | The composable interface |
| {mod}`bartorch.learning`, {mod}`bartorch.interop` | Adapters to networks and to DeepInverse | Yes | PyTorch around the composable interface |

A command is the shortest route to a standard reconstruction, and its result
is BART's by construction.  The composable interface is needed when the
encoding has no BART application — an additional factor in the forward model,
a subspace with a field correction, an operator defined in Python — when a
solver is called from an outer loop such as Gauss-Newton, or when gradients
are required.

## Commands and apps

{func}`bartorch.tools.pics` runs BART's `pics`.  {func}`bartorch.apps.pics`
performs the same steps with this package's objects: the sampling pattern,
the modulation into BART's uncentred convention and the data scaling in
Python, then an encoding from {mod}`bartorch.linop` under a solver from
{mod}`bartorch.optim`.  On a Cartesian grid the two return identical tensors;
along a trajectory they agree to floating-point round-off, because FINUFFT
accumulates over threads in an order that varies between runs.  An app is the
starting point for a variant of an application: its steps are Python that can
be read and changed.  {func}`bartorch.apps.mobafit` is the exception to
agreement: it fits a TorchSim signal model rather than BART's, and returns
named maps in physical units.

The `bartorch` command line reads a `bart` command line.  Where an app exists
for the command and expresses every option given, the app runs; otherwise the
command itself runs.  {func}`bartorch.cli.route` reports which.

## Underneath the interfaces

| Layer | Role |
| --- | --- |
| Python and ctypes | A tensor crosses as its data pointer and its shape reversed into a BART dimension vector; a C-order tensor and a BART array of the reversed dimensions are the same memory, so operators copy nothing.  Commands work on copies of their inputs by default, because some BART commands write into their inputs. |
| bartorch C ABI (`libbartorch`) | Runs commands and operators under BART's error handler, so an error or a failed assertion inside BART raises {class}`~bartorch.BartError` instead of ending the process.  Arrays BART allocates are allocated by PyTorch on the device of the call. |
| Embedded BART | The commands, the linear and nonlinear operators and the iterative algorithms, compiled from the pinned submodule without source changes. |
| Encoding executor | bartorch's implementation of the MRI encoding operators: SENSE over an FFT, a NUFFT or a wave transform, applied to a slab of coils at a time and built from BART's operators; described in {doc}`encoding`. |
| Substituted backends | Components compiled in place of BART's: FINUFFT and cuFINUFFT for every non-uniform Fourier transform and point spread function ({doc}`non-cartesian`); MKL's DFTI or the compiled pocketfft for the FFT; BLAS and LAPACK routines from MKL, from PyTorch's linked library, or from SciPy. |

On a CUDA device, operators and solvers work on device memory directly; BART
uses cuFFT and cuBLAS, cuFINUFFT computes non-uniform transforms, and BART's
work is ordered against PyTorch's current stream.  Some commands allocate host
temporaries internally, and are given host copies of their inputs; their
results are returned on the device.

## Why the arithmetic is BART's

bartorch computes what BART computes.  The substitutions change how a
transform or a matrix routine is evaluated, not the reconstruction built on
it; the iterations of {mod}`bartorch.optim` reproduce BART's step sizes,
penalty updates and stopping rules, and the test suite holds the Cartesian
solvers to BART's own results bit for bit.  Where a configuration cannot be
served — a non-Cartesian transform FINUFFT cannot compute, a term an iteration
cannot apply — the call raises an error with the reason rather than computing
the result by another method.
