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

FINUFFT is a dependency, not an extra: `pip install bartorch` brings it on
every platform it ships a wheel for, and nothing has to be asked for.

```bash
pip install "bartorch[cufinufft]"    # a transform on a card
pip install "bartorch[finufft]"      # only where FINUFFT ships no wheel:
                                     # Linux on aarch64, an Intel Mac.  pip
                                     # then builds it from source.
```

```python
import bartorch
import bartorch.tools as bt

image = bt.pics(kspace, maps, t=traj)   # FINUFFT underneath, unasked
```

BART builds every non-Cartesian transform through `nufft_create`, and that is
the seam: with FINUFFT in place, `nufft`, `pics`, `nlinv` and `moba` compute
their forward and adjoint transforms with FINUFFT's type 2 and type 1 instead
of BART's Kaiser-Bessel gridder and oversampled FFT. Nothing about kernels or
deapodisation has to agree, because FINUFFT does the whole transform. A
transform BART runs on a card is served by cuFINUFFT, one on the host by
FINUFFT; an operator that is asked for both holds a plan on each side.

The normal operator stays BART's: `A^H A` is a convolution, and `pics` applies
it as one multiply against the point spread function BART already knows how to
compute, so a solve gets FINUFFT's transforms and BART's Toeplitz embedding
together. On a 256x256 eight-coil radial dataset of 401 spokes:

| | transform | adjoint | `pics` |
| --- | --- | --- | --- |
| BART | 70 ms | 130 ms | 1.66 s |
| FINUFFT | 26 ms | 26 ms | 1.06 s |

and the reconstruction agrees with BART's to 1.2e-03 while sitting an order of
magnitude closer to an explicit discrete Fourier sum. `--no-toeplitz` means
what it means on both.

A trajectory that varies across frames and a subspace basis go through it too:
the frames join the point set rather than splitting it, so one plan over the
raveled trajectory serves every coefficient, and the basis contracts them away
on the k-space side of the transform pair.

Nothing asks for any of it. The substitution puts itself in place the first
time anything needs one, and what it cannot serve -- weights that do not lie
along k-space, images that vary along an axis the trajectory varies on -- is an
error naming the reason rather than a quieter answer from BART's own gridder:
an order further from the transform and several times slower, with nothing to
say so. `bartorch.finufft.decline_reason()` is that reason, and
`operators_built()` and `normals_built()` count what was built.

```python
A = LinearOperator.nufft(traj, (8, 1, 256, 256), (8, 401, 256, 1), basis=basis)
```

is the same transform as an operator, for chaining and solving outside BART's
tools, and it takes the weights and the subspace basis for the same reason the
tools do. `bartorch.finufft.configure(tolerance=1e-3)` is what makes a large
three-dimensional problem fit. Both wheels ship a compiled library, so this is
a pip install and nothing more.

## How it is built

- `csrc/include/bartorch.h` is a plain C ABI. Nothing compiled links against
  Python or torch, so one wheel per platform serves every interpreter and
  every torch version, and the extension is reached through ctypes.
- Every BART translation unit is compiled unchanged from the submodule. A
  BART routine is replaced by leaving its unit out and compiling one with the
  same signatures: the in-memory array registry allocates through the host,
  and CBLAS, LAPACKE and the FFTW interface forward to Fortran-ABI and DFTI
  tables filled at import from the compiled BLAS, LAPACK and MKL the process
  already holds. Where a machine has no MKL, which is macOS, the transform
  compiled into the library serves instead.
- BART's nested functions become Blocks under clang and heap trampolines under
  GCC 14+, so the library loads without an executable stack either way. Both
  compilers are tested in CI.
- Only the `bartorch_*` symbols are exported.

## CUDA

```bash
pip install -e . --config-settings=cmake.define.BARTORCH_CUDA=ON
```

A release also carries a prebuilt CUDA wheel. It has the same name, version
and platform tag as the CPU wheel, which PyPI cannot hold twice, so it is
attached to the GitHub release rather than published:

```sh
pip install https://github.com/mcencini/bartpy/releases/download/<tag>/bartorch-<version>-py3-none-manylinux_2_28_x86_64.whl
```

It carries BART's device code for five architectures and links the CUDA
runtimes torch already brings, so it costs a few megabytes and needs no
toolchain on the target.

A tensor on a card selects that card, and BART's streams are ordered against
torch's current stream by an event in each direction, so neither side
synchronises it.

An operator takes the memory as it is: BART's linops reach their arguments
through `md_` operations, which dispatch on where a pointer is, so a device
tensor is transformed where it lies. A tool is given host memory, because
BART's tools are command mains that map their inputs the way the command line
does and several read them there; the tensors cross to the host and the result
crosses back, and the work in between is BART's own device path — the one `-g`
selects on the command line — on the card the tensors came from.

```python
import bartorch

bartorch.cuda.available()        # built with CUDA, and a device present
bartorch.cuda.set_streams(2)     # overlap BART's transfers with its kernels
bartorch.cuda.use_memcache(False)  # an operator gives its device memory straight back
```

The CUDA runtime libraries are linked dynamically from the `nvidia` wheels
torch already depends on, so a CUDA build adds about 2 MB of device code
rather than hundreds of megabytes of vendored libraries.

## Status

Linux and macOS on the host, and a CUDA build walked through on an RTX 4060 by
`scripts/check_device.py`: tools and operators on the card, cuFINUFFT serving
its transforms, `pics` with and without the Toeplitz normal. Windows and the
remaining solver entry points are next; see `AGENTS.md`.

## License

MIT. BART is distributed under its own BSD license; see `bart/LICENSE`.
pocketfft is BSD-3 and BlocksRuntime is MIT; both are vendored under
`third_party/` with their licenses.
