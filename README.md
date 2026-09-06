# bartorch

The [Berkeley Advanced Reconstruction Toolbox (BART)](https://mrirecon.github.io/bart/),
embedded in a Python process and driven on `torch.Tensor` objects.

Every BART tool is a function. Every BART operator is an object that applies
to tensors, chains with other operators and goes to BART's solvers. An
operator written in Python enters BART the same way, so a torch signal model
is fitted by BART's Gauss-Newton solver and a Python normal operator runs
inside BART's conjugate gradients.

## Install

```bash
pip install bartorch
```

The wheel carries one C library holding all of BART and depends on `numpy` and
`torch` alone. BLAS and LAPACK are taken from the library torch already loaded
into the process; the FFT is pocketfft. No BART binary, no MKL, no FFTW.

From source, with clang and CMake:

```bash
git clone --recurse-submodules https://github.com/mcencini/bartpy
cd bartpy
pip install -e .
```

## Tools

```python
import bartorch.tools as bt

kspace = bt.phantom([256, 256], kspace=True, ncoils=8)
maps = bt.ecalib(kspace, calib_size=24, maps=1)
image = bt.pics(kspace, maps, R="W:7:0:0.005")
```

Shapes are C order, so the last axis is the one BART calls the first, and
wherever BART takes a bitmask the Python function takes axis indices:
`bt.fft(x, axes=(-1, -2))`. A tool's output is the tensor BART wrote, allocated
by torch through the library's allocator callback.

## Operators

```python
from bartorch.ops import LinearOperator, NonlinearOperator

S = LinearOperator.multiply_sum(maps, (1, 256, 256), (8, 256, 256))
F = LinearOperator.fft((8, 256, 256), axes=(-1, -2))
A = F @ S                                   # S first, then F
x = A.lstsq(kspace, lambda_=1e-3)           # BART's conjugate gradients

N = LinearOperator.nufft(traj, (8, 256, 256), toeplitz=True)
y = N(coil_images)
xn = N.adjoint(y)
```

A Python operator with a forward and an adjoint is a BART linop; a torch
function is a BART nlop whose derivative and adjoint come from autograd:

```python
M = NonlinearOperator.from_torch(signal_model, (2, 256, 256), (echoes, 256, 256))
x = (F @ M).irgnm(kspace, x0, iterations=10)     # model-based reconstruction
```

`LinearOperator.from_callbacks(oshape, ishape, forward, adjoint, normal=...)`
takes a normal operator such as an mrtoeplitz kernel and lets BART use it in
place of the forward-adjoint pair.

## How it is built

- `csrc/include/bartorch.h` is a plain C ABI. Nothing compiled links against
  Python or torch, so one wheel per platform serves every interpreter and
  every torch version, and the extension is reached through ctypes.
- Every BART translation unit is compiled unchanged from the submodule. A
  BART routine is replaced by leaving its unit out and compiling one with the
  same signatures: the in-memory array registry allocates through the host,
  the FFTW interface is served by pocketfft, and CBLAS and LAPACKE forward to
  a Fortran-ABI table filled at import from whatever the process provides.
- The library is a clang build: BART's nested functions become Blocks, so it
  loads without an executable stack.
- Only the `bartorch_*` symbols are exported.

## Status

CPU, Linux and macOS. CUDA, Windows, FINUFFT gridding and the remaining
solver entry points are in progress; see `AGENTS.md` for the design.

## License

MIT. BART is distributed under its own BSD license; see `bart/LICENSE`.
pocketfft is BSD-3 and BlocksRuntime is MIT; both are vendored under
`third_party/` with their licenses.
