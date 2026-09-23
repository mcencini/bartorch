[![Tests](https://github.com/mcencini/bartorch/actions/workflows/tests.yml/badge.svg)](https://github.com/mcencini/bartorch/actions/workflows/tests.yml)
[![Docs](https://github.com/mcencini/bartorch/actions/workflows/docs.yml/badge.svg)](https://mcencini.github.io/bartorch/)
[![Lint](https://github.com/mcencini/bartorch/actions/workflows/lint.yml/badge.svg)](https://github.com/mcencini/bartorch/actions/workflows/lint.yml)
[![Typos](https://github.com/mcencini/bartorch/actions/workflows/typos.yml/badge.svg)](https://github.com/mcencini/bartorch/actions/workflows/typos.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/pypi/v/bartorch.svg)](https://pypi.org/project/bartorch/)
[![Downloads](https://img.shields.io/pypi/dm/bartorch.svg)](https://pypistats.org/packages/bartorch)
[![Python](https://img.shields.io/pypi/pyversions/bartorch.svg)](https://pypi.org/project/bartorch/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Wheels](https://img.shields.io/badge/wheels-Linux%20x86--64%20%7C%20macOS%20arm64-2f6f9f)](https://github.com/mcencini/bartorch/blob/main/.github/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f6f9f.svg)](https://github.com/mcencini/bartorch/blob/main/LICENSE)
[![Source](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/mcencini/bartorch)
[![Stars](https://img.shields.io/github/stars/mcencini/bartorch?style=flat&logo=github&color=2f6f9f)](https://github.com/mcencini/bartorch/stargazers)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mcencini/bartorch/main/docs/_static/bartorch-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/mcencini/bartorch/main/docs/_static/bartorch-logo.svg" alt="bartorch" width="440">
  </picture>
</p>

bartorch embeds BART, the Berkeley Advanced Reconstruction Toolbox, in the
Python process and calls it on PyTorch tensors in host or CUDA device memory.
BART's commands are Python functions; its linear and nonlinear operators are
composable Python objects; its iterative algorithms are solver classes and
iteration blocks.  Applying an operator or a solver to a tensor that
requires a gradient records it for PyTorch autograd.  The arithmetic is BART's
except where bartorch substitutes a component: FINUFFT and cuFINUFFT compute
every non-uniform Fourier transform, the FFT and BLAS/LAPACK routines come from
MKL, PyTorch's linked library or SciPy, and the MRI encoding operators run on
bartorch's own executor built from BART's operators.

## Features

- BART commands (`pics`, `ecalib`, `nlinv`, `moba`, ...) as functions of
  tensors, and the `bartorch` command line, which accepts the arguments of `bart`.
- MRI encoding operators — Cartesian, non-Cartesian and wave-encoded SENSE,
  off-resonance correction — composed with `@` and `+` into single BART operators.
- BART's regularization terms and its CG, IST, FISTA, ADMM and primal-dual
  iterations, as solvers and as differentiable iteration blocks.
- Nonlinear operators, iteratively regularized Gauss-Newton, and quantitative
  signal models from TorchSim.
- Adapters for unrolled networks, plug-and-play denoisers and DeepInverse.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mcencini/bartorch/main/docs/_static/architecture-dark.svg">
    <img src="https://raw.githubusercontent.com/mcencini/bartorch/main/docs/_static/architecture.svg" alt="bartorch architecture: PyTorch tensors enter the command-style and composable interfaces, which call the bartorch C ABI through ctypes; the ABI runs the embedded BART, whose non-uniform Fourier transforms and linear algebra are served by substituted backends" width="760">
  </picture>
</p>

## Quick start

```bash
pip install bartorch
```

```python
import bartorch.tools as bt
from bartorch import priors

kspace = bt.phantom(128, coils=8, kspace=True)  # (coils, z, y, x)
maps = bt.ecalib(kspace, maps=1)
image = bt.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005), solver="fista")
```

The same reconstruction can be assembled from an encoding operator and a
solver; [Interfaces and execution](https://mcencini.github.io/bartorch/latest/explanation/execution-model.html)
describes which interface fits which task.

## Documentation

<https://mcencini.github.io/bartorch/> has the
[user guide](https://mcencini.github.io/bartorch/latest/guides/user/index.html)
(installation, supported platforms, data layout), the
[developer guide](https://mcencini.github.io/bartorch/latest/guides/developer/index.html),
conceptual [explanations](https://mcencini.github.io/bartorch/latest/explanation/index.html),
executed [examples](https://mcencini.github.io/bartorch/latest/examples/index.html)
and the [API reference](https://mcencini.github.io/bartorch/latest/api/index.html),
with a version switcher between the development version and the releases.

## Citation

bartorch has no publication or archival DOI.  Work that uses it should cite
BART and the methods it applied (ESPIRiT, compressed sensing, nonlinear
inversion, ...), and report the bartorch version or commit and the pinned
BART revision; [Contributors and citation](https://mcencini.github.io/bartorch/latest/misc/contributors.html)
gives the references and what else a reproducible report records.

## License

bartorch is MIT-licensed.  The embedded BART (BSD-3-Clause) and the vendored
pocketfft (BSD-3-Clause) and BlocksRuntime (MIT or NCSA) keep their own licenses; see
[License and third-party notices](https://mcencini.github.io/bartorch/latest/misc/license.html).
bartorch is an independent project, not affiliated with or endorsed by the BART
developers, the PyTorch Foundation or The Linux Foundation.  The logo combines
BART's mark with the PyTorch logo's flame; PyTorch, the PyTorch logo and any
related marks are trademarks of The Linux Foundation.
