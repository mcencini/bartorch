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
| `csrc/host_reads.c` | The entry points BART reads element by element, answered over a host copy when a tool is on a card. |
| `csrc/finufft.c`, `nufft_finufft.c` | FINUFFT's and cuFINUFFT's entry points, and BART's NUFFT operator built out of a pair of their plans. |
| `csrc/compat/` | The `cblas.h`, `lapacke.h` and `fftw3.h` BART includes. |
| `third_party/` | pocketfft and BlocksRuntime, vendored with their licenses. |
| `src/bartorch/` | The package: `_lib.py` (ctypes), `_backend.py` (which library serves BLAS and LAPACK), `_buffer.py` (a tensor over one of BART's buffers, host or device), `core/graph.py` (tools on tensors), `ops.py` (operators), `finufft.py` (the substitution), `tools/` (one function per BART command). |
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
of fatbinary for five architectures on top of the host library.
`CUDA_GET_CUDA_DEVICE_NUM` switches BART to asking the driver whether a
pointer is on a device, which is what lets a torch CUDA tensor be passed to an
operator without being registered anywhere. Ordering is two events per call
rather than a synchronise: `bartorch_cuda_wait_for_stream` holds BART's
streams until torch's queued work has run and `bartorch_cuda_signal_stream`
does the reverse.

A tensor on a card selects that card for the length of the call, which is what
`bartorch.cuda.ordered` does, and that is also what turns BART's own device
path on: `bart_use_gpu` is what `-g` sets on the command line, so passing `-g`
to a tool as well changes nothing.

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

Density weights are a diagonal in k-space -- BART multiplies the transform by
them on the way out and by their conjugate on the way back -- so the operator
carries them itself, which is what lets `pics` take this path: it always
passes a sampling pattern. Weights that do not lie along k-space are declined,
and `bartorch_nufft_decline_text` says why; the counters say whether an
operator was built by FINUFFT or by BART, which is how a test asserts that a
tool ran on it rather than that FINUFFT was merely available.

**A decline is an error, not a quieter reconstruction.** A caller who asked
for FINUFFT and silently got BART's gridder would get an answer an order of
magnitude further from the transform and several times slower, with nothing
to say so. So `nufft_create2` refuses, naming the reason, and
`enable(fallback=True)` is what hands those back to BART. `enable` itself
raises when `finufft` is missing, and when `cufinufft` is missing on a machine
whose card BART would otherwise use.

**A trajectory that varies across frames is one plan, not one per frame.**
Every axis the trajectory indexes is a sample of one transform and the rest
are separate transforms, so frames join the readout and the spokes in the
point set rather than splitting it. That is what makes a subspace fit: each
coefficient is transformed over the same raveled trajectory, and the basis is
a contraction on the k-space side of the pair -- `md_ztenmul` on the way out,
its conjugate on the way back, which is what `nufft.c` does either side of its
own gridder. The normal stays BART's, now built over the basis as well, so a
subspace `pics` still solves against a point spread function.

BART hands one of two k-spaces in -- `nufft` gives it a single coefficient and
`pics` gives it all of them -- so the operator carries both: `out_dims` is what
the caller sees, `grd_dims` what the transform pair works in. FINUFFT executes
on a transform's samples together, and BART's own order is already that unless
a sample axis sits above a batch axis, which frames do; a buffer in the
transform's layout stands between when it does. An image that varies along a
sample axis is declined, because that would need a transform per frame rather
than one plan over all of them.

The entry points that read the operator's internals -- `nufft_get_psf*`,
`nufft_update_*`, `nufft_precond_create` -- refuse on one of these rather than
read the wrong struct. `pics` only reaches them for `--psf_export` and
`--psf_import`.

A plan holds the coordinates by pointer rather than copying them, so each side
owns its own arrays and they are freed with the operator; the trajectory in
radians is kept once on the host, and a side copies it to wherever its plans
are.

**The normal operator stays BART's.** A^H A is a convolution, so a solve
applies it as one multiply against a point spread function rather than a
transform each way -- and `nufft.c` already computes that function, with
`compress_psf`, `decomposed_psf` and `lowmem` around it. So the substituted
operator asks `bart_nufft_create2` for BART's own operator over the same
trajectory and borrows its normal, while FINUFFT keeps the pair. Nothing is
reimplemented and nothing is added to the dependency list. `conf.toeplitz`
decides, so `pics --no-toeplitz` and `nufft -t` mean what they mean, and
`bartorch.finufft.normals_built()` says which of the two answered.

On a 256x256 eight-coil radial dataset of 401 spokes, against an explicit
discrete Fourier sum on one spoke:

| | forward | adjoint | error |
| --- | --- | --- | --- |
| BART | 70 ms | 130 ms | 3.6e-05 |
| FINUFFT, tolerance 1e-6 | 26 ms | 26 ms | 2.2e-06 |
| FINUFFT, tolerance 1e-4 | 16 ms | 16 ms | 1.1e-05 |

and `pics` over the same data, agreeing with BART's own reconstruction to
1.2e-03:

| | Toeplitz | forward and adjoint |
| --- | --- | --- |
| BART | 1.66 s | 6.44 s |
| FINUFFT | 1.06 s | 2.33 s |

**cuFINUFFT is the same table.** `csrc/finufft.c` holds two of them, filled
from the `finufft` and `cufinufft` wheels; without the `cufinufft` wheel a
transform BART would run on a card stays with BART's own operator rather than
quietly running on the host.

Which table serves a transform is decided by where its arguments are, not by
where the trajectory is, because BART hands one operator memory on either
side: `pics` takes its first adjoint from the k-space it mapped and then
iterates on device vectors. So the operator holds a side per place -- a pair
of plans, the coordinates they point at, and the weights -- and builds one the
first time a transform is asked for there. Everything around the transform --
taking a trajectory component, rescaling it into radians, the scaling, the
weights -- goes through BART's own `md_` operations, which is what makes one
piece of code serve both. A callback reaches a device buffer through the CUDA
array interface (`src/bartorch/_buffer.py`), which is how a Python operator
sees one.

The cuFINUFFT wheel is taken through the C API it exports from 2.3 on: an
int64 sample count in `cufinufftf_setpts`, and defaults filled from the
options struct alone, spelled `cufinufft_default_opts` without the precision
suffix FINUFFT uses.

`LinearOperator.finufft` is the same transform reached without BART's tools,
for chaining and solving in Python. It makes a plan once and reuses it, matches
BART's sign and scaling, and goes into BART's solvers unchanged. A BART
trajectory always carries three components, so whether a transform is two- or
three-dimensional is decided by whether kz is used, not by the trajectory's
shape.

**A tool keeps the card where that is safe; an operator always does.** Two
things stand between a BART tool and the memory it was handed. The few entry
points that read an array element by element rather than through `md_` --
`estimate_im_dims` sizing an image from a trajectory, `estimate_scaling_norm`
taking a median of k-space -- are answered in `csrc/host_reads.c` over a host
copy of that one array, which is what BART already does for its own virtual
pointers. What cannot be reached that way is a tool that allocates a temporary
of its own on the host and mixes it with its input: `pocsense` takes its
pattern from `md_alloc`, `nlinv` from `anon_cfl`, `ecalib` sets `bart_use_gpu`
from its own flag. `md_` operations take the host path unless every argument
is on a device, and take it silently, so that is a segmentation fault rather
than a slower answer.

Which tools are which is a property of their own code, so `_ON_DEVICE` in
`core/graph.py` holds only what has been run on a card and checked against the
host, and a test in `tests/test_cuda.py` runs every name in it. The rest are
given host memory and their result comes back on the card. On a 256x256
eight-coil radial dataset that is `pics` in 0.12 s rather than 0.25 s and
`fft` in 3 ms rather than 10 ms.

Whichever way a tool runs, what BART allocates for itself comes from torch on
the memory that tool was given: an output on the other side of the bus from
its input is the same silent host path. BART also maps input files
copy-on-write and some tools write into them, so a tensor is cloned unless the
caller turns that off.

An operator reaches its arguments through BART's `md_` operations, which
dispatch on where a pointer is, so it takes a device tensor as it stands and
never writes its input: that path is zero-copy in both directions.

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

The compiler's own runtime is linked statically on Linux, because otherwise
the toolchain's floor becomes the target system's: a GCC 14 build asks
`libgcc_s` for `GCC_14.0.0`, which no released distribution ships, and that is
a `dlopen` failure rather than a fallback. The library exports a C ABI and
exchanges no C++ objects with the process it is loaded into, so its libgcc and
libstdc++ can be its own. What is left is glibc, which the manylinux image
sets, and libgomp. A wheel built this way asks for `GLIBC_2.28` and nothing
else.

**Errors.** Every library entry point runs under BART's error catcher, so
`error()` inside BART returns an error code and its message, captured
through `vendor_log`, and never exits the process. A BART tool's text
output arrives the same way.

That includes assertions, which is how BART checks the arguments a caller is
most likely to get wrong. BART routes `assert` through `error()` only under
`USE_DWARF`, which also wants libdw and libunwind for backtraces; without it
glibc's `assert` calls `abort()` and a wrong shape takes the interpreter down.
`csrc/api.c` answers `__assert_fail` instead, which needs neither the define
nor the libraries, and the symbol is hidden so it binds inside this library
alone.

## Commands

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
BARTORCH_LIBRARY=$PWD/build/libbartorch.so PYTHONPATH=src pytest tests/
pip install -e .                    # the same through scikit-build-core
pip install -e . --config-settings=cmake.define.BARTORCH_CUDA=ON   # with device code
python scripts/check_device.py      # everything a card can answer that a host cannot
python build_tools/gen_tools.py     # after a submodule bump
ruff format src tests build_tools && ruff check src tests build_tools
```

## Tests

A numerical test pins BART against something outside BART: numpy's FFT, an
explicit discrete Fourier sum, a closed-form signal, an adjoint identity. A
test that compares BART to BART proves nothing.

## On a card

`scripts/check_device.py` walks the device path in dependency order, each
check independent and naming its own reason, so the first failure is the thing
to fix. On an RTX 4060 Laptop, CUDA 12.8, all nine pass: `fft` against numpy
to 9e-08, BART's own NUFFT against an explicit discrete Fourier sum to 1.4e-03
and cuFINUFFT's to 5.7e-07, the device and host transforms agreeing to
3.4e-06, and `-g` changing nothing that the device pointers had not already
decided.

On a 256x256 eight-coil radial dataset of 401 spokes, best of five, `pics`
takes 0.46 to 0.57 s on the card with the point spread function and 0.46 to
0.53 s with the transform pair -- each inside the other's scatter -- against
1.9 to 2.6 s and 5.0 to 5.9 s on the same machine's host. The pair is cheap
enough on a card that halving the number of transforms stops being worth
measuring; on the host the point spread function is still worth 2.5x. This is
a 45 W laptop card, so a run measured cold and one measured after the clocks
have dropped differ by more than the two normals do.

More than one BART stream makes no difference that measurement can resolve:
one, two and four streams reconstruct in the same half second, and the spread
between them is smaller than the spread between repetitions of any one of
them. What BART holds on the card is what `bartorch.cuda.use_memcache` decides
for an operator -- 57 MB against nothing on that dataset -- and nothing at all
for a tool, because BART's `main` clears the cache when a command ends.

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

Windows, tools with optional extra outputs, and the wider solver surface
(ADMM, FISTA, proximal operators) through the operator layer.

A tool that takes device memory as it stands. BART guards the host reads that
would break -- `estimate_im_dims` copies to the host when it is handed one --
for its own virtual pointers, not for a raw device pointer, so the route in is
`csrc/memcfl.c` handing BART a `vptr_wrap_cfl` rather than the pointer itself,
and then finding out which of BART's guards are complete.

Two of mrtoeplitz's ideas have no route in from here: a transfer that stays on
the host and is streamed across in chunks, and bfloat16 transfers, which halve
what crosses the bus. Both are decisions about how the point spread function
is stored, which lives inside `nufft.c`, so neither is reachable by
substituting an entry point -- they would need a BART edit or a normal
operator written here. The seam is one function: `toeplitz_for` in
`csrc/nufft_finufft.c` decides what the operator's normal is, and an
mrtoeplitz kernel behind a host callback would go there. What BART does have
is `compress_psf`, `decomposed_psf` and `lowmem`, and its own overlap:
`bartorch.cuda.set_streams` sets `cuda_num_streams`, which is what puts BART's
transfers and its arithmetic on different streams.

Routing BART's device allocations through torch's allocator is a further step:
`mem_device_malloc` takes the allocator as a parameter, so replacing
`num/mem.c` would do it without a BART edit.
