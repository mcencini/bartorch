# bartorch

bartorch runs BART, the Berkeley Advanced Reconstruction Toolbox, inside the
Python process on torch tensors: BART's applications as functions, its linear
and nonlinear operators as composable objects, and its iterative solvers as
classes.  It computes nothing BART does not, except where it substitutes a
faster implementation: FINUFFT for every non-Cartesian transform, and a SENSE
operator that applies its coils in batches.

## Install

```bash
pip install bartorch
pip install "bartorch[mkl]"          # Linux: MKL for BART's BLAS, LAPACK and FFT
pip install "bartorch[cufinufft]"    # non-Cartesian transforms on a CUDA device
```

Wheels are built for Linux x86_64 and macOS on Apple silicon; elsewhere pip
builds from source, which needs clang or GCC 14+ and CMake.  The
[installation guide](docs/guides/user/installation.md) covers CUDA builds and
platform notes.

## Where to start

| Task | Start here |
| --- | --- |
| Reconstruct with a BART application | `bartorch.tools`: `pics`, `nlinv`, `ecalib` |
| Build an encoding and solve it | `bartorch.linop` and `bartorch.optim` |
| Transform, filter or register arrays | the functions in `bartorch` |
| Fit a torch signal model | `bartorch.nlop` and `bartorch.optim.IRGNM` |

## Quickstart

```python
import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, prox

kspace = bt.phantom(128, coils=8, kspace=True)        # (8, 1, 128, 128)
maps = bt.ecalib(kspace, maps=1)

# BART's own application ...
image = bt.pics(kspace, maps, regularizers="W:3:0:0.005", solver="fista")

# ... or the same problem assembled from an operator and a solver.
# pics also scales the data first; see optim.data_scaling.
A = linop.Sense(maps.squeeze(1), (8, 128, 128))
x = optim.FISTA(prox.Wavelet((-1, -2), 0.005), maxiter=50)(kspace, A)

spectrum = bartorch.fft(bt.phantom(128), axes=(-2, -1), unitary=True)
```

Shapes are C order, so the last axis is the one BART calls the first, and
wherever BART takes a bitmask a function here takes axis indices.  BART
bitmasks survive only inside strings passed through to BART, such as the
`regularizers` of `pics`.

## License

MIT.  BART is distributed under its own BSD license (`external/bart/LICENSE`);
pocketfft (BSD-3) and BlocksRuntime (MIT) are vendored under `external/` with
their licenses.
