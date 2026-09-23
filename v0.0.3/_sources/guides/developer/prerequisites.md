# Development prerequisites

| Tool | Requirement |
| --- | --- |
| Git | With submodule support; BART is the submodule `external/bart` |
| Python | 3.10 or later, with PyTorch installed first as described in {doc}`../user/installation` |
| C and C++ compiler | clang, or GCC 14 or later.  BART's nested functions are compiled as Blocks by clang and as heap trampolines (`-ftrampoline-impl=heap`) by GCC 14; CMake rejects older GCC at configuration |
| CMake | 3.18 or later, with Ninja or Make |
| OpenMP | The runtime matching the compiler, for example `libomp-dev` with clang on Debian and Ubuntu.  Without it the build succeeds and runs single-threaded |
| CUDA (optional) | A CUDA toolkit with `nvcc` for `-DBARTORCH_CUDA=ON`; a CUDA build of PyTorch does not include a compiler |

FINUFFT is a runtime dependency that `pip install -e .` installs.  When the
suite is run from `src/` without installing, install it separately
(`pip install finufft`): without it the non-Cartesian tests fail rather than
skip, because the substitution declining is an error.  `pip install mkl
deepinv` enables the tests that need MKL and the DeepInverse adapter.

Windows is not a target: BART does not build on it, and WSL2 is used as a Linux
environment.
