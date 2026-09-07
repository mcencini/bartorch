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
| `csrc/cuda.c` | Device selection, stream ordering against the caller's stream, and BART's memory cache. Present in both builds; the CPU build reports that it has no CUDA. |
| `csrc/finufft.c`, `nufft_finufft.c` | FINUFFT's entry points, and BART's NUFFT operator built out of a pair of its plans. |
| `csrc/compat/` | The `cblas.h`, `lapacke.h` and `fftw3.h` BART includes. |
| `third_party/` | pocketfft and BlocksRuntime, vendored with their licenses. |
| `src/bartorch/` | The package: `_lib.py` (ctypes), `_backend.py` (which library serves BLAS and LAPACK), `core/graph.py` (tools on tensors), `ops.py` (operators), `finufft.py` (the substitution), `tools/` (one function per BART command). |
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
every interpreter and torch version, and it is why a device pointer and a
stream cross the same ABI as a host pointer.

**BLAS, LAPACK and FFT come from compiled libraries in the process, never
from Python.** At import, `_backend.py` fills the routine table from, in
order: MKL when the `mkl` extra is installed, the library torch links, and
SciPy's `cython_blas` and `cython_lapack`, whose capsules hold the addresses
of the compiled OpenBLAS routines. A test asserts that every routine resolved
to a library and that no LAPACK entry fell back to the built-in reference.

MKL is preferred because it is the only one that covers all thirty routines —
torch links MKL statically and exports the thirteen it calls itself — and
because it is faster where it counts. On a 256x256 eight-coil dataset:

| | ecalib | pics | svd 512 |
| --- | --- | --- | --- |
| MKL | 0.13 s | 0.08 s | 0.04 s |
| torch + SciPy | 0.29 s | 0.08 s | 0.04 s |

The gap is all in `ecalib`, whose per-voxel eigendecompositions are the
LAPACK-heavy part; `pics` is FFT-bound and does not care. On small arrays MKL
loses instead, because it brings its own OpenMP runtime beside BART's and two
thread pools cost more than MKL saves there. Neither build showed a duplicate
OpenMP runtime error; `MKL_THREADING_LAYER=GNU` is the escape hatch if one
appears.

There is no MKL wheel for macOS, so a Mac gets Accelerate through torch and
SciPy's OpenBLAS for the rest. `BARTORCH_BLAS_LIBRARY` puts one source first:
`mkl`, `torch`, `scipy`, or a path, which is also how to benchmark one against
another.

On a device none of this applies: BART calls cuBLAS directly, and it has no
GPU LAPACK, so eigendecompositions and SVDs come back to the host table.

The FFT is pocketfft on the host, the same code torch uses on CPU without MKL,
and cuFFT on a device.

**CUDA is the same ABI.** `-DBARTORCH_CUDA=ON` compiles BART's thirteen `.cu`
files with nvcc and links cudart, cuFFT and cuBLAS dynamically, which are the
three BART uses, so the wheel carries device code and nothing else: about 2 MB
of fatbinary for five architectures on top of the host library. `CUDA_GET_CUDA_DEVICE_NUM`
switches BART to asking the driver whether a pointer is on a device, which is
what lets a torch CUDA tensor be passed in without being registered, and the
allocator callback returns tensors on whichever device the caller selected.
Ordering is two events per call rather than a synchronise:
`bartorch_cuda_wait_for_stream` holds BART's streams until torch's queued work
has run and `bartorch_cuda_signal_stream` does the reverse.

**FINUFFT arrives the same way MKL does.** The `finufft` and `cufinufft`
wheels each carry a compiled shared library with a plain C plan API, so they
are a pip extra and nothing is built or vendored. `csrc/finufft.c` holds the
entry points and `_finufft.py` hands them over along with the byte offset of
FINUFFT's options struct, read from the same package so a release that moves a
field cannot silently corrupt it.

**Underneath BART's own tools the seam is `nufft_create`, not the gridder.**
`nufft.c` is compiled with `nufft_create`, `nufft_create2`, `nufft_get_psf*`
and `nufft_update_*` renamed, and `precond.c` with `nufft_precond_create`
renamed, so BART's own operator survives as `bart_nufft_*` and
`csrc/nufft_finufft.c` answers to the original names. What it returns is a
`linop_create` whose forward is FINUFFT's type 2 with a negative exponent and
whose adjoint is type 1 with a positive one, both scaled by one over the square
root of the voxel count. Nothing about gridding kernels or deapodisation has to
be matched, because FINUFFT does the whole transform, and every tool that
builds a NUFFT gets it: `nufft`, `pics`, `nlinv`, `moba`.

Density weights are a diagonal in k-space, so they are chained on as a
`linop_cdiag_create`, which is what lets `pics` take this path -- it always
passes a sampling pattern. A subspace basis, weights that do not lie along
k-space, and a trajectory that varies across frames are declined, and
`bartorch_nufft_decline_reason` says which; the counters say whether an
operator was built by FINUFFT or by BART, which is how a test asserts that a
tool ran on it rather than that FINUFFT was merely available.

The entry points that read the operator's internals -- `nufft_get_psf*`,
`nufft_update_*`, `nufft_precond_create` -- refuse on one of these rather than
read the wrong struct. `pics` only reaches them for `--psf_export` and
`--psf_import`.

A plan holds the trajectory by pointer rather than copying it, so the
coordinate arrays live in the operator's data and are freed with it.

On a 256x256 eight-coil radial dataset of 401 spokes, against an explicit
discrete Fourier sum on one spoke:

| | forward | adjoint | error |
| --- | --- | --- | --- |
| BART | 70 ms | 130 ms | 3.6e-05 |
| FINUFFT, tolerance 1e-6 | 26 ms | 26 ms | 2.2e-06 |
| FINUFFT, tolerance 1e-4 | 16 ms | 16 ms | 1.1e-05 |

`pics` end to end comes out about the same either way, because BART's operator
answers a normal-equations apply with one point-spread-function multiply and
this one does not: it has no PSF, so conjugate gradients runs a forward and an
adjoint per iteration. Giving it a Toeplitz normal operator is the obvious
next thing.

`LinearOperator.finufft` is the same transform reached without BART's tools,
for chaining and solving in Python. It makes a plan once and reuses it, matches
BART's sign and scaling, and goes into BART's solvers unchanged. A BART
trajectory always carries three components, so whether a transform is two- or
three-dimensional is decided by whether kz is used, not by the trajectory's
shape.

**Tools copy their inputs; operators do not.** BART maps input files
copy-on-write and some tools write into them, so a tool gets a clone unless
the caller turns that off. An operator never writes its input, so the
operator path is zero-copy in both directions.

**Compilers.** BART is GNU C, and the difficulty is its nested functions.
Under clang they become Blocks, resolved by the vendored runtime on Linux and
by libSystem on macOS. Under GCC they become trampolines, and a trampoline on
the stack needs an executable stack, which glibc 2.41 refuses to `dlopen`; so
GCC 14's `-ftrampoline-impl=heap` is required and older GCC is rejected at
configure time. BART's own `NOEXEC_STACK` workaround does not help here: it
parses a trampoline layout GCC emits only for non-PIC executables, not for a
shared library. Both compilers are built and tested in CI.

MSVC cannot compile BART. It does not have to: nothing here is a Python
extension module, so a Windows build is a plain DLL that ctypes loads
whatever compiler torch was built with. That DLL comes from clang or
mingw-w64 GCC 14, and BART already carries the `src/win/` shims (`mmap`,
`fmemopen`) such a build needs.

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
BART takes a bitmask, Python takes axis indices. A flag's value can be an
array rather than a number -- `pics(kspace, maps, t=traj)`, `-p` for a
sampling pattern, `-B` for a basis -- and is registered and copied like any
other input. BART's `fft` tool is
unnormalised unless asked for the unitary form; the `LinearOperator.fft`
operator is unitary. `nufft` output is scaled by one over the grid side per
transformed axis pair, with a negative exponent.

Write for someone reading the code as it is now. No text about what the code
used to be. A docstring carries what a caller needs: one line, Parameters,
Returns, Raises.

## What is not done

Windows, a Toeplitz normal operator for the FINUFFT NUFFT, cuFINUFFT on the
device path, tools with optional extra outputs, and the wider solver surface
(ADMM, FISTA, proximal operators) through the operator layer.

The CUDA path is verified only as far as a machine without a card allows: it
compiles, links, loads, reports no device, and runs the whole host suite. The
tests that need a card are written and skip. What wants checking on real
hardware, in order: that a tool on device tensors returns a device tensor with
the right numbers, that BART's allocations and torch's caching allocator
coexist on a card with little memory (`bartorch.cuda.use_memcache(False)`
gives BART's memory straight back), whether tools need `-g` passed as well as
`bart_use_gpu` being set, and whether more than one BART stream actually
overlaps transfer with arithmetic. Routing BART's device allocations through
torch's allocator is a further step: `mem_device_malloc` takes the allocator
as a parameter, so replacing `num/mem.c` would do it without a BART edit.
