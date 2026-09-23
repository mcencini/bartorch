# Installation

## PyTorch installation

Install the PyTorch build for the target device first, with the command the
[PyTorch installation selector](https://pytorch.org/get-started/locally/)
gives for the CPU or for a CUDA version.  bartorch requires PyTorch 2.2 or
later; installing bartorch into an environment without PyTorch installs the
default build from PyPI.

## bartorch installation

```bash
python -m pip install bartorch
```

This installs the wheel for the platform where one exists (see
{doc}`prerequisites`) together with NumPy, SciPy, FINUFFT, TorchSim and
MRI-NUFFT.  A wheel contains the compiled library with BART embedded; no BART
installation and no compiler are needed.  Optional components are installed as
extras:

```bash
python -m pip install 'bartorch[mkl]'        # Linux x86-64: MKL for BLAS, LAPACK and FFT
python -m pip install 'bartorch[cufinufft]'  # non-uniform FFTs of CUDA tensors
python -m pip install 'bartorch[deepinv]'    # bartorch.interop.to_deepinv
```

The installation is checked by building a phantom:

```python
import bartorch
import bartorch.tools as bt

print(bartorch.__version__, bartorch.bart_version())
print(bartorch.build_info())
print(bartorch.backend_sources())
image = bt.phantom(32)
print(image.shape, image.dtype, image.device)
```

{func}`bartorch.build_info` reports the compiler, the OpenMP and CUDA
configuration of the library, and {func}`bartorch.backend_sources` the library
serving each BLAS and LAPACK routine and the FFT: MKL when the `mkl` extra is
installed, otherwise the routines PyTorch links, and SciPy's for the rest.

(source-builds)=
## Source builds

Where no wheel exists, pip builds the source distribution, which needs:

| Tool | Requirement |
| --- | --- |
| C and C++ compiler | clang, or GCC 14 or later; BART's nested functions are compiled as clang Blocks or as GCC heap trampolines, and older GCC is rejected at configuration |
| CMake | 3.18 or later |
| OpenMP | The compiler's OpenMP runtime, for example `libomp-dev` with clang on Debian and Ubuntu; without it BART runs single-threaded |

The compiler is selected with `CC` and `CXX`:

```bash
CC=clang CXX=clang++ python -m pip install bartorch --no-binary bartorch
```

A checkout of the repository is built as described in the
{doc}`developer guide <../developer/installation>`.

## CUDA

CUDA support requires a CUDA build of both PyTorch and bartorch;
`torch.cuda.is_available()` and {func}`bartorch.cuda_available` report each.
The PyPI wheel is the CPU build.  The CUDA build of each version, a Linux
x86-64 wheel with the same file name, is attached to the
[GitHub release](https://github.com/mcencini/bartorch/releases) of that
version:

```bash
python -m pip install https://github.com/mcencini/bartorch/releases/download/<tag>/<wheel>
```

It contains device code for compute capabilities 7.5, 8.0, 8.6, 8.9 and 9.0
and links the CUDA 12 runtime, cuFFT and cuBLAS dynamically, which a CUDA 12
build of PyTorch provides.  A source build with CUDA passes
`-C cmake.define.BARTORCH_CUDA=ON` to pip and needs the CUDA toolkit with `nvcc`.

Operators and commands run on the device that holds their tensor arguments.
Operators work on device memory directly; some commands allocate host
temporaries internally and are given host copies of their inputs, with their
results returned on the device.

## Non-Cartesian backends

Every non-uniform Fourier transform, in the commands and in the operators, is
computed by FINUFFT for tensors in host memory and by cuFINUFFT for tensors on
a CUDA device.  BART's own gridding implementation is not used.

| Transform | Backend | Requirement | When unavailable |
| --- | --- | --- | --- |
| Host tensors | FINUFFT | Installed as a dependency | The transform raises an error |
| CUDA tensors | cuFINUFFT | The `cufinufft` extra | On a machine with a CUDA device and a CUDA build of bartorch, the substitution is not installed and every non-uniform transform, on the host as well, raises an error until cuFINUFFT is installed |
| A configuration FINUFFT cannot serve | None | None | The transform raises {class}`~bartorch.BartError` with the reason |

{doc}`../../explanation/non-cartesian` describes the transform, its tolerance
and the configurations that are refused.

(macos-openmp)=
## macOS OpenMP compatibility

The PyTorch and FINUFFT wheels for macOS each contain a copy of the LLVM
OpenMP runtime (`torch/lib/libomp.dylib` and `finufft/.dylibs/libomp.dylib`).
The runtime terminates the process when a second copy initializes
(`OMP: Error #15`).

The first time a process needs a non-uniform transform, before it loads
FINUFFT's library, bartorch changes the OpenMP load command of
`libfinufft.dylib` to PyTorch's copy with `install_name_tool`, re-signs the
library with an ad hoc signature (`codesign`), and checks that the load command
changed.  One runtime is then loaded and shared by PyTorch and FINUFFT.

| Condition | Behaviour |
| --- | --- |
| Both copies are LLVM's `libomp` with compatible versions, the Xcode Command Line Tools are installed, and the `finufft` package directory is writable | The library is modified once; later processes find it already modified |
| Any of these does not hold | The library is not modified, and non-uniform transforms raise an error stating the reason; importing bartorch and all other functions are unaffected |
| `finufft` is reinstalled or upgraded | The original library is restored and the modification is repeated at the next non-uniform transform |

In a source checkout, `python scripts/macos_openmp.py diagnose` reports what
was found and what would be done, `patch` applies the modification, and
`verify` checks the result.  `KMP_DUPLICATE_LIB_OK=TRUE` is neither set nor
recommended: it lets two copies of the runtime run in one process, a
configuration the LLVM OpenMP runtime does not support.

## Windows and WSL2

BART does not build on Windows, and bartorch has no Windows build.  Under WSL2,
bartorch is installed as on Linux.

## Command-line interface

Installation provides the `bartorch` command, which accepts the command lines
of BART's `bart` executable and operates on CFL files:

```sh
bartorch pics -l1 -r0.01 -i30 kspace sensitivities image
bartorch --list
```

A script written for `bart` runs with the command name replaced, or with
`alias bart=bartorch`; {doc}`../../api/cli` describes which commands run as
{mod}`bartorch.apps` pipelines.
