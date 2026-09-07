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

The wheel carries one C library holding all of BART, and depends on `numpy`,
`scipy` and `torch`. BLAS and LAPACK are compiled routines already in the
process, so no numerical work is ever done in Python.

```bash
pip install "bartorch[mkl]"     # Linux and Windows: MKL for every routine
```

MKL is worth the extra where it exists. It is the only source covering all
thirty routines BART calls — torch links MKL statically and exports the
thirteen it uses itself — and on a 256x256 eight-coil dataset an ESPIRiT
calibration takes 0.13 s against 0.29 s, the difference being its per-voxel
eigendecompositions. There is no MKL wheel for macOS, where Accelerate and
SciPy's OpenBLAS serve instead. `BARTORCH_BLAS_LIBRARY` puts one source first:
`mkl`, `torch`, `scipy`, or a path.

From source, with clang or GCC 14+ and CMake:

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

## FINUFFT

```bash
pip install "bartorch[finufft]"      # host
pip install "bartorch[cufinufft]"    # device
```

```python
N = LinearOperator.finufft(traj, (8, 256, 256))
```

It computes the same operator as `LinearOperator.nufft` and is interchangeable
with it, so it chains and solves the same way — on a 256x256 radial trajectory
it takes 10.7 ms against BART's 19.2 ms. Both wheels ship a compiled library,
so this is a pip install and nothing more.

BART's own tools do not use FINUFFT yet. The machinery to put it underneath
them is in place — `grid2`, `grid2H` and the deapodisation are interceptable,
with BART's gridder kept as the fallback — but BART grids onto several
image-sized arrays rather than one oversampled one, and FINUFFT's kernel is
sized in cells of the array it is handed, so the two geometries do not yet
agree. `install_gridder()` checks itself against BART's gridder and declines
rather than leave a mismatched kernel in place; see `AGENTS.md`.

## How it is built

- `csrc/include/bartorch.h` is a plain C ABI. Nothing compiled links against
  Python or torch, so one wheel per platform serves every interpreter and
  every torch version, and the extension is reached through ctypes.
- Every BART translation unit is compiled unchanged from the submodule. A
  BART routine is replaced by leaving its unit out and compiling one with the
  same signatures: the in-memory array registry allocates through the host,
  the FFTW interface is served by pocketfft, and CBLAS and LAPACKE forward to
  a Fortran-ABI table filled at import from the compiled BLAS and LAPACK the
  process already holds.
- BART's nested functions become Blocks under clang and heap trampolines under
  GCC 14+, so the library loads without an executable stack either way. Both
  compilers are tested in CI.
- Only the `bartorch_*` symbols are exported.

## CUDA

```bash
pip install -e . --config-settings=cmake.define.BARTORCH_CUDA=ON
```

A tool or operator given CUDA tensors runs on that device. BART recognises the
pointers, writes its output into a tensor torch allocated on the same device,
and its streams are ordered against torch's current stream by an event in each
direction, so nothing crosses the host and neither side synchronises the card.

```python
import bartorch

bartorch.cuda.available()        # built with CUDA, and a device present
bartorch.cuda.set_streams(2)     # overlap BART's transfers with its kernels
bartorch.cuda.use_memcache(False)  # give freed memory back for torch to use
```

The CUDA runtime libraries are linked dynamically from the `nvidia` wheels
torch already depends on, so a CUDA build adds about 2 MB of device code
rather than hundreds of megabytes of vendored libraries.

## Status

Linux and macOS on the host, and a CUDA build that compiles, links and runs
the host suite — the device path itself is written but has not been run on a
card. Windows, FINUFFT underneath BART's own tools, and the remaining solver
entry points are next; see `AGENTS.md`.

## License

MIT. BART is distributed under its own BSD license; see `bart/LICENSE`.
pocketfft is BSD-3 and BlocksRuntime is MIT; both are vendored under
`third_party/` with their licenses.
