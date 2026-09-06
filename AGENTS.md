# bartorch, for an agent working on it

bartorch embeds BART, the Berkeley Advanced Reconstruction Toolbox, in a
Python process and drives it on torch tensors. The compiled part is a plain
C library with a small C ABI; Python reaches it through ctypes.

## The shape of the repository

| Path | What is in it |
| --- | --- |
| `bart/` | BART, as a git submodule, compiled unchanged. |
| `csrc/include/bartorch.h` | The C ABI. The only header a host sees. Plain C: no complex types, no variable-length arrays. |
| `csrc/api.c` | Command execution under BART's error catcher, log capture, threads. |
| `csrc/memcfl.c` | The in-memory array registry, replacing `bart/src/misc/memcfl.c`. Arrays BART creates come from the host's allocator callback. |
| `csrc/fftw_pocketfft.cpp` | The FFTW guru interface BART plans with, executed by pocketfft. |
| `csrc/backend.[ch]`, `ref_blas.c`, `cblas_shim.c`, `lapacke_shim.c` | CBLAS and LAPACKE as BART calls them, forwarded to a table of Fortran-ABI routines with reference BLAS as the fallback. |
| `csrc/ops.c` | Operators: host callbacks as BART linops and nlops, BART's own operators as handles, least squares and Gauss-Newton. |
| `csrc/compat/` | The `cblas.h`, `lapacke.h` and `fftw3.h` BART includes. |
| `third_party/` | pocketfft and BlocksRuntime, vendored with their licenses. |
| `src/bartorch/` | The package: `_lib.py` (ctypes), `_backend.py` (which library serves BLAS and LAPACK), `_linalg.py` (NumPy LAPACK callbacks), `core/graph.py` (tools on tensors), `ops.py` (operators), `tools/` (one function per BART command). |
| `build_tools/gen_tools.py` | Generates `tools/_generated.py` from the BART sources. |
| `attic/prototype/` | An earlier pybind11 extension, kept for reference and not built. |

## Design rules

**No BART edit.** Whatever BART needs changed is done by leaving a
translation unit out of the build and compiling one with the same
signatures, or by a compile definition. Moving the submodule to a newer BART
should be a pointer bump plus regenerating the tool wrappers.

**No torch or Python in the compiled code.** The library exports only the
`bartorch_*` ABI. Torch owns every tensor; the library sees data pointers and
BART-order dimension vectors. That is what makes one wheel per platform serve
every interpreter and torch version, and it is why the CUDA path, when it
comes, will pass device pointers and streams through the same ABI.

**BLAS, LAPACK and FFT come from the process.** At import, `_backend.py`
looks for Fortran-ABI routines in the library torch loaded (`libtorch_cpu`
exports the MKL routines torch itself uses), then in the process, then
serves what is missing with NumPy callbacks. The FFT is pocketfft, the same
code torch uses on CPU without MKL. Nothing is downloaded beyond torch.

**Tools copy their inputs; operators do not.** BART maps input files
copy-on-write and some tools write into them, so a tool gets a clone unless
the caller turns that off. An operator never writes its input, so the
operator path is zero-copy in both directions.

**Compiler.** BART is GNU C and the library is a clang build: nested
functions become Blocks (the vendored runtime on Linux, libSystem on macOS),
so the library loads without an executable stack. GCC would need one: BART's
static-chain workaround reads a trampoline layout GCC emits only for non-PIC
executables, not for a shared library. MSVC cannot build BART; a Windows
wheel will be a clang build too.

**Errors.** Every library entry point runs under BART's error catcher, so
`error()` inside BART returns an error code and its message, captured
through `vendor_log`, and never exits the process. A BART tool's text
output arrives the same way.

## Commands

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
BARTORCH_LIBRARY=$PWD/build/libbartorch.so PYTHONPATH=src pytest tests/
pip install -e .                    # the same through scikit-build-core
python build_tools/gen_tools.py     # after a submodule bump
ruff format src tests build_tools && ruff check src tests build_tools
```

## Tests

A numerical test pins BART against something outside BART: numpy's FFT, an
explicit discrete Fourier sum, a closed-form signal, an adjoint identity. A
test that compares BART to BART proves nothing.

## Conventions

Shapes are C order; a BART dimension vector is the reversed shape. Wherever
BART takes a bitmask, Python takes axis indices. BART's `fft` tool is
unnormalised unless asked for the unitary form; the `LinearOperator.fft`
operator is unitary. `nufft` output is scaled by one over the grid side per
transformed axis pair, with a negative exponent.

Write for someone reading the code as it is now. No text about what the code
used to be. A docstring carries what a caller needs: one line, Parameters,
Returns, Raises.

## What is not done

CUDA (device pointers through the ABI, BART's GPU kernels, cuFINUFFT
gridding, stream ordering against torch), Windows, FINUFFT replacing BART's
gridder, tools with optional extra outputs, and the wider solver surface
(ADMM, FISTA, proximal operators) through the operator layer.
