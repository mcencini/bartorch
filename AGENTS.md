# bartorch, for an agent working on it

bartorch embeds BART, the Berkeley Advanced Reconstruction Toolbox, in a
Python process and drives it on torch tensors. The compiled part is a plain
C library with a small C ABI; Python reaches it through ctypes.

## The shape of the repository

Four directories at the top, and a file belongs to whichever it is: `src/` is
everything written here, `external/` everything that is not, `tests/` and
`docs/` the rest.  Nothing of ours sits outside `src/`, and nothing of anyone
else's sits inside it.

| Path | What is in it |
| --- | --- |
| `external/bart/` | BART, as a git submodule, compiled unchanged. |
| `external/pocketfft/`, `external/blocksruntime/` | Vendored with their licenses. |
| `src/bartorch/` | The Python package. |
| `src/csrc/` | The compiled library. |
| `scripts/` | Everything a developer runs by hand: the two generators, the docs build, the suite, the device check. |
| `cmake/` | What the build system runs and a person does not. |
| `tests/`, `docs/` | The rest. |

`src/csrc/` is three things, and a file belongs to whichever it is: `abi/` is
the boundary -- what the host calls, and what BART's environment asks of the
host in return; `ops/` is what the host builds and drives; `substitute/` is
what runs in BART's place, both its own computations and the libraries it
would otherwise have been linked against.

| Path | What is in it |
| --- | --- |
| `src/csrc/include/bartorch.h` | The C ABI. The only header a host sees. Plain C: no complex types, no variable-length arrays. |
| `src/csrc/compat/` | The `cblas.h`, `lapacke.h` and `fftw3.h` BART includes. |
| `src/csrc/abi/api.c` | Command execution under BART's error catcher, log capture, threads. |
| `src/csrc/abi/memcfl.c` | The in-memory array registry, replacing `external/bart/src/misc/memcfl.c`. Arrays BART creates come from the host's allocator callback. |
| `src/csrc/abi/cuda.c` | Device selection, stream ordering against the caller's stream, and BART's memory cache. Present in both builds; the CPU build reports that it has no CUDA. |
| `src/csrc/abi/host_reads.c` | The entry points BART reads element by element, answered over a host copy when a tool is on a card. |
| `src/csrc/ops/ops.c` | Operators: host callbacks as BART linops and nlops, BART's own operators as handles, least squares and Gauss-Newton. |
| `src/csrc/ops/sense.c` | The one encoding executor, parameterised by the form: the coil slab loop, the streaming of a bank held as kernels, the contraction over terms, and `bartorch_encoding_operator` behind them, with the CUDA kernels beside it that the streamed normal is made of. |
| `src/csrc/ops/grid.c`, `grid.cuh` | The transforms on a grid a slab carries: the pattern-and-basis kernel, a table of sampled phase encodes, the wave front, and the normal through cuFFT's callbacks. |
| `src/csrc/ops/iter.c` | The solve `pics` runs -- `italgo_config`, `lsqr2` -- over an operator and terms the host assembled. |
| `src/csrc/substitute/fft.cpp` | The FFTW guru interface BART plans with, executed by MKL where the process has it. |
| `src/csrc/substitute/backend.[ch]`, `ref_blas.c`, `cblas_shim.c`, `lapacke_shim.c` | CBLAS and LAPACKE as BART calls them, forwarded to a table of Fortran-ABI routines with reference BLAS as the fallback. |
| `src/csrc/substitute/finufft.c`, `nufft_finufft.c` | FINUFFT's and cuFINUFFT's entry points, and BART's NUFFT operator built out of a pair of their plans -- and the normal, which stores one of those in BART's operator through `noncart/nufft_priv.h` rather than letting it grid one. |
| `src/csrc/substitute/psf.c` | The three `compute_psf*` entry points, so that the adjoint transform a point spread function is comes from the substitution. |
| `src/bartorch/` | The package.  Public: the functions in `fourier.py`, `wavelet.py`, `thresh.py`, `util.py`, `interp.py` and `_settings.py`, re-exported flat as `bartorch.*`; `linop/` and `nlop/` (a class per operator); `optim/` (a class per BART iteration); `priors/` (BART's regularization terms, and its denoisers); `tools/` (BART's applications, in five sections); `io.py` (CFL files); `interop.py` (the deepinv adapter).  Private: `_abi.py` (the ctypes signatures, generated from the header), `_lib.py` (finding and loading the library), `_marshal.py` (what an ABI argument looks like), `_backend.py` (which library serves BLAS and LAPACK), `_buffer.py` (a tensor over one of BART's buffers, host or device), `_dispatch.py` (running a command on tensors), `_operator.py` (what every operator shares), `_grid.py` (what the operations on a grid share, including BART's motion layout), `_finufft.py` and `_cuda.py` (the substitution's and the card's controls), `_catalogue.py` and `_options.py` (what BART declares, and what each option is called here), `_call.py` (the mark on a hand-written wrapper, and wrappers built from the catalogue), `_coverage.py` (where each command is exposed, or why not), `_macos_openmp.py` (pointing FINUFFT's OpenMP runtime at torch's, so one is loaded); inside `linop/`, `form.py` (the encoding form and the plan it reports) and `plan.py` (matching a composition against that form). |
| `scripts/gen_abi.py` | Generates `_abi.py` from `src/csrc/include/bartorch.h`. Run after changing the header; `tests/test_abi.py` fails when the checked-in file is not what it writes. |
| `scripts/gen_catalogue.py` | Generates `_catalogue.py` from the BART sources: every command, its arguments, and every option with both spellings. Run after a submodule bump. |
| `scripts/run_tests.sh` | Builds whatever changed on the C side, then runs the suite against `src/`, without installing. |
| `scripts/sources.py` | What the library is built from, in one place, so the script and `tests/test_build.py` cannot disagree about it. |
| `scripts/lint.sh` | Ruff over `src/` and `tests/`, which is what the lint workflow runs; `--fix` writes. |
| `scripts/benchmark_encodings.py` | The encoding timings `docs/design/composed-encodings.md` records, one case per process, each printing the plan it was lowered into beside its times. |
| `scripts/benchmark_newton.py` | The Gauss-Newton timings `docs/design/nonlinear-fusion.md` records, with the encoding applied as its normal and as a pair. |
| `scripts/macos_openmp.py` | `bartorch._macos_openmp` by hand: `diagnose` reads, `patch` rewrites and re-signs, `verify` proves it. The substitution does it for itself on first use, so this is for seeing what it found and for an environment where it could not. |
| `scripts/build_docs.sh` | Builds the reference the way the workflow does. |
| `scripts/check_device.py` | Everything a card can answer that a host cannot, in dependency order. |
| `cmake/embed.cmake` | Writes a file's bytes into a C array, for the LTO-IR the CUDA build links. |
| `attic/prototype/` | An earlier pybind11 extension, kept for reference and not built. |

Every BART command is wrapped by hand, derived from the catalogue into a `tools` section, or
private with a reason in `_coverage.py`; `tests/test_tools.py` holds the partition, so a command
a BART update adds has to be placed.  `tests/test_docs.py` holds every public name to a place in
`docs/api/`.

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

The FFT is planned through the FFTW guru interface and executed by MKL's DFTI,
filled from the same table, which on Linux is torch's own MKL and needs nothing
installed. MKL's FFTW interface is not used: it refuses more than
one loop dimension and BART passes one per dimension it is not transforming, so
DFTI takes the transformed axes and the longest loop axis and `src/csrc/substitute/fft.cpp`
walks whatever is left. It is given one thread: MKL is fast enough serially to
beat a threaded pocketfft, and a thread team of its own is a second OpenMP
runtime spinning against BART's, which inside a tool costs several times more
than the transform gains. Where no source has DFTI, which is macOS, and for a
description DFTI declines, the transform compiled into the library serves
instead. On a device it is cuFFT, through BART's own `fft-cuda.c`.

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
`bartorch._cuda.ordered` does, and that is also what turns BART's own device
path on: `bart_use_gpu` is what `-g` sets on the command line, so passing `-g`
to a tool as well changes nothing.

**FINUFFT is a dependency; cuFINUFFT is an extra.** The `finufft` and
`cufinufft` wheels each carry a compiled shared library with a plain C plan
API, so nothing is built or vendored either way. `src/csrc/substitute/finufft.c` holds the
entry points and `_finufft.py` hands them over along with the byte offset of
FINUFFT's options struct, read from the same package so a release that moves a
field cannot silently corrupt it.

The two are not optional in the same way. FINUFFT *is* the NUFFT here -- every
non-Cartesian transform goes through the substitution under `nufft_create` --
so a bartorch without it does non-Cartesian work slowly rather than well,
which is a dependency and not a choice. cuFINUFFT serves a transform on a
card, and most machines have no card, so it stays an extra.

**On macOS the two wheels have to be made one runtime first.** torch carries
an OpenMP runtime and the FINUFFT wheel carries its own, and LLVM's runtime
ends the process rather than run beside a second copy of itself (`OMP: Error
#15`).  `_finufft.openmp_runtimes()` reads the loaded images for that pair
before the first call into FINUFFT; more than one and the substitution
declines, `install_once` says so at warning level, and every non-Cartesian
transform is then refused.  BART's gridder does not quietly take over: an
answer an order further from the transform and several times slower, arriving
with nothing to say so, is worse than no answer.

`_macos_openmp.ensure()` is the fix, and `use_in_tools` runs it before it
loads FINUFFT's library -- which is the only moment it can, because loading
that library is what brings its runtime into the process.  The two copies are
the same runtime: both LLVM's libomp, both compatibility version 5.0.0, and
every OpenMP symbol `libfinufft.dylib` imports is exported by the copy torch
carries.  So FINUFFT's library is pointed at torch's with `install_name_tool`
and re-signed -- not optional on Apple Silicon, where changing a load command
invalidates the signature -- one runtime is loaded, and the substitution
installs as it does everywhere else.

There is no pip post-install hook for a wheel, so this is the nearest thing:
the first NUFFT repairs the install, and the next `pip install -U finufft`
undoes it and the one after that repairs it again.  It refuses rather than
patching where the two are not one runtime, checks afterwards that the load
command really changed -- `codesign` can report success over a file that did
not -- and raises nothing, because an environment where the file cannot be
rewritten should refuse transforms rather than fail to import.  Where it could
not, the refusal carries its reason and `scripts/macos_openmp.py diagnose`
prints what it saw.

`KMP_DUPLICATE_LIB_OK=TRUE` is not this: the flag tells one runtime to
tolerate a second live copy and is documented by its own authors as unsafe;
here there is one copy, and torch and FINUFFT share its pool.  The macOS CI
job reads the pair with `diagnose`, brings the library up -- which is where
`ensure()` runs, as it would on anyone's machine -- and then checks with
`verify` that the loader agrees, all before the suite, so a repair that did
not work lands there rather than inside a test.  Linux is not asked: its
loader resolves the duplicate instead of dying on it.

The requirement carries no marker, and that is a decision about which wheels
exist rather than an oversight. FINUFFT ships none for Linux on aarch64 -- it
never has -- and dropped the Intel Mac after 2.4.0; its sdist wants CMake,
ninja, a C++ compiler and a fetched FFTW. A bartorch wheel for a platform
FINUFFT has no wheel for could only either install something that cannot do
non-Cartesian work, or start a source build for someone who asked for a wheel.
So no such wheel is built: `publish.yml` builds Linux x86_64 and macOS arm64,
which are platforms FINUFFT ships wheels for too, and aarch64 is served by the
sdist -- where compiling BART is already the price of entry, and compiling
FINUFFT beside it costs nothing new.

There is no `finufft` extra. An old `pip install 'bartorch[finufft]'` still
installs FINUFFT, and pip warns that the extra is not provided, which is the
right thing to hear. `tests/test_dependencies.py` holds all of this: that the
requirement is there, that it is unconditional, that no extra shadows it, and
that the metadata pip was actually given promises what pyproject does.
`_finufft.required_but_missing()` is the message for an absent one, and says
it is a broken install rather than a choice.

**Underneath BART's own tools the seam is `nufft_create`, not the gridder.**
`nufft.c` is compiled with `nufft_create`, `nufft_create2`, `nufft_get_psf*`
and `nufft_update_*` renamed, and `precond.c` with `nufft_precond_create`
renamed, so BART's own operator survives as `bart_nufft_*` and
`src/csrc/substitute/nufft_finufft.c` answers to the original names. What it returns is a
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

**Nothing reaches BART's gridder without having been sent there.** An answer
an order of magnitude further from the transform and several times slower,
arriving with nothing to say so, is worse than no answer. So the substitution
installs itself the first time anything needs it -- a caller who has the
package does not have to ask -- and `nufft_create2` refuses whatever it cannot
serve, naming the reason. The substitution being switched off is a reason like
any other, which is what closes the case that used to be silent: the library
as it starts, before anyone has mentioned FINUFFT.

BART's own gridder is not reachable from the package's surface at all, and
nothing in the library opens it: `bartorch_nufft_allow_fallback` is set by
`_finufft.barts_own_gridder()` and by `use_in_tools(False)`, which are the
agreement check and the tests, and by nothing else.  Every way the
substitution can fail to install -- `finufft` missing, `cufinufft` missing on
a machine with a card, two OpenMP runtimes, the agreement check disagreeing --
leaves it closed, so what follows is a refusal naming the reason.  There is
nothing a caller can pass to end up on the gridder. `_finufft.use_in_tools` raises when `finufft` is missing
-- which on a platform it ships a wheel for means the install has lost it --
or when `cufinufft` is missing on a machine whose card BART would otherwise
use. A test that enters that block would carry it into the next test, so
`tests/conftest.py` puts the substitution back after every one.

The operator layer says what the tools say: `linop.NUFFT` takes the
weights and the subspace basis, because the normal is a point spread function
over both and a chain could not be. Anything it cannot express is another way
back to BART, so there is no separate FINUFFT operator beside it -- one
NUFFT that is FINUFFT's underneath, the way the tools are.

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

**A point spread function is an adjoint transform of ones.** `nufft.c` builds
one by taking the adjoint NUFFT of ones over a doubled trajectory, and reaches
that transform through its own `nufft_create2`, which the rename sends to
BART's gridder along with everything else in that file. So `compute_psf`,
`compute_psf2` and `compute_psf2_decomposed` are renamed too and `src/csrc/substitute/psf.c`
answers to them: the same squared weights and basis, the same doubled grid,
shifts and decomposition, with the transform in the middle being whichever
`nufft_create2` answers. `nlinv`, `moba`, `rtnlinv`, `noir/model2` and the
`psf` tool call these directly.

What that is worth is in the numbers. On the un-doubled grid of a 16x16
twelve-spoke trajectory, against the sum the function is defined by, BART's
own point spread function is 2.5e-02 out and this one is 6.9e-07 -- its
tolerance. On the doubled grid, where the function is properly sampled, the
two agree to 6e-04.

The decomposed one takes a set of frequencies at a time, each with its own
shifted trajectory and its own image. That is a stack of transforms rather
than one plan over frames, so it is always built the way BART builds it for
`lowmem`, which also holds one set at a time.

**The normal stays BART's convolution over a function computed here.**
`nufft.c` would compute its own from inside the file where the rename cannot
reach it, so `toeplitz_for` turns that off: `conf.nopsf` is the switch
`pics --psf_import` uses to bring a function in from outside, and with it set
BART grids nothing. What is left is to make the function -- `compute_psf2`,
whose transform is the substitution's -- and to store it the way the operator
wants, which `src/csrc/substitute/nufft_finufft.c` does over the dimensions the operator
worked out for itself, through `noncart/nufft_priv.h`.

Everything BART does with the function afterwards is still BART's, and every
way of storing it still works and costs no gridding:

| `--nufft-conf` | what it stores |
| --- | --- |
| (none), `lowmem`, `no-precomp` | the whole complex function |
| `decomposed-psf` | a set of frequencies at a time |
| `upper-triag-psf` | half of a Hermitian one, for a subspace |
| `real-psf` | its real part, as floats |
| `compress-psf` | the entries that are not zero, beside an index of where they were |

`operators_built()` returning zero for BART is the whole claim, and every
route to one of BART's operators increments it.

The mask a compressed function keeps is spread with FINUFFT's kernel too.
BART finds it by spreading the sampling pattern with its own, and that is the
wrong footprint once the function is FINUFFT's: what the mask has to cover is
where this function has signal. `spreadinterponly` is the spreading with
nothing after it -- no transform, no deapodisation -- so what comes back is
the kernel's own reach, on whichever grid it is asked for. Both wheels carry
the field, spelled differently and at different offsets, which the options
layout reads from each package like the others.

It is asked for the grid the mask lives on, one set of frequencies at a time:
the doubled grid decomposes into that many copies of the image, each carrying
the samples shifted by its own fraction of a cell, and a point any of them
reaches is a point the mask keeps. Doing it a set at a time is what keeps the
doubled grid from ever being allocated, which is the whole reason the
decomposition is there.

It cannot come off the function instead, which is the obvious thing to try.
A transfer function is nonzero over the whole grid: the adjoint transform
crops in image space after an oversampled FFT, which convolves in k-space,
and the deapodisation does not undo it. Measured on a 32x32 grid of 48
spokes, the fraction of it standing above a millionth of its peak is 100 per
cent, for BART's own gridder and for FINUFFT alike. So the mask is not a
question about magnitudes but about geometry -- which Cartesian frequencies
the trajectory reaches -- which is what a spreading answers and a threshold
does not.

The width it spreads with is the one the transform's kernel really covers.
`ns` is FINUFFT's name for its kernel width, counted in cells of the fine grid
it spreads on, and BART's `-w` is the same number for its own kernel. A kernel
of `ns` cells on a grid oversampled by sigma covers `ns/sigma` cells of the
grid underneath, and that is the footprint the mask needs -- BART says the
same thing as a width of K/2 at os 1 for a transform of width K at os 2. That
arithmetic is this library's own; what `ns` is, is FINUFFT's.

So `ns` is asked for rather than worked out. FINUFFT sizes its kernel from the
tolerance and the upsampling by a formula in its own `src/common/kernel.cpp`,
which it exports only as a C++ symbol over an internal struct, and which
cuFINUFFT does not export at all: a copy of it here would be a copy that goes
stale quietly. `fi_measure_width` spreads one sample with `spreadinterponly`
and counts what lands, which is the kernel, and the two libraries are asked
separately because nothing says they must agree. Going the other way -- the
tolerance that buys a width, which `-w` needs and so does the mask -- is a
bisection over the same question, with the published formula as a starting
guess and never as the answer.

The probe is a grid of 32 per transformed axis, made and freed per question,
and it has to carry the real number of dimensions: `ns` depends on them.
At a thousandth on a grid a quarter over it is 5 in one dimension and in two,
and 6 in three -- which is this library's own default, so a probe in one
dimension would answer the wrong question for a volume.

Both happen when an operator is built and never while one is applied: a
reconstruction of ten conjugate-gradient iterations and one of sixty make the
same eight FINUFFT plans, sixteen with a compressed function and eleven with a
width asked for.

The mask is planned with the tolerance and upsampling the operator was planned
with, not with the library's defaults, or a caller who set `-o` or `-w` would
get a mask for a kernel that is not the one they asked for.
FINUFFT refuses an upsampling of one and takes no width, so the width is asked
for as the tolerance that buys it: `fi_width` and `fi_tolerance_for` are that
formula both ways round, checked against what the library plans for every
tolerance from a tenth to a millionth at both upsamplings.

What it costs against BART's mask, in norm, from compressing:

| | 64^2, 64 spokes | 64^2, 128 | 128^2, 128 | 128^2, 256 |
| --- | --- | --- | --- | --- |
| this mask | 1.1e-02 | 5.1e-03 | 2.8e-03 | 1.4e-03 |
| BART's | 2.5e-02 | 1.5e-02 | 5.8e-03 | 3.3e-03 |

Two to three times less, everywhere the problem is posed well enough for the
comparison to mean anything. Far below that -- sixteen spokes across a 128
grid -- conjugate gradients wander and the difference between two
reconstructions says more about the conditioning than about either mask.

The oversampling of two the Toeplitz embedding needs is the grid, not the
kernel: `compute_psf2` asks for an image of 2N and the decomposition rewrites
that as 2^d grids of N, and FINUFFT's upsampling sizes its own fine grid
underneath, which is free. So the two can be set apart, and are. On a 32-grid
the function comes back as 4 sets of 32x32 -- (2N)^2 points -- at a quarter
over, at twice, at a tolerance of a hundredth, and from BART; what the
upsampling changes is how close it is, 8.9e-04 at a thousandth against
3.6e-03 at a hundredth. This is the same object `torchkbnufft` builds at
twice its image size whatever its `grid_size` is.

Accuracy is not what says the mask is in the right place: one that keeps the
wrong points but more of them reconstructs well too. The compression rate is
what says it. Given FINUFFT the kernel BART uses -- an upsampling of two and
a tolerance that buys ns = 6, covering the 3 cells BART's width of 6 at an
oversampling of 2 does -- the two masks keep 86 and 84 per cent of a 64-grid
of 64 spokes, 83 and 82 of a 128-grid of 128, 82 and 80 of a 128-grid of 48,
and 86 apiece on a card. Within rounding a width to whole cells, which is
where this one is the wider.

The same width off a different upsampling keeps the same points, which is the
other half of the check and the half that does not need BART at all: a
millionth at an upsampling of two and a thousandth at a quarter over both buy
a width of four, and both keep 88 per cent of that grid. A width of three
keeps 86 and a width of six keeps 91, so it is monotone in the width, as
nested masks have to be. A displaced one would not land there.

Only one kernel can be compared against BART, and it is BART's own. Its
`compute_psf_nufft_conf` starts from `nufft_conf_defaults`, so it computes its
point spread function at a width of six on a grid twice over whatever `-w` and
`-o` say -- and its Kaiser-Bessel table is one table for the process, so any
other width in the same command dies on `Kaiser-Bessel window initialized with
different beta`. BART's Toeplitz path is width-six-only by its own
construction.

Which is why the operator borrowed for the normal is asked for BART's own
grid and width and nothing else: the embedding needs the grid twice over,
anything else sends `nufft_create2` down a chained path that is not even the
same data underneath, and none of BART's kernel is evaluated anyway. Only the
transforms follow what was configured, which is the point.

Nothing here reuses a configuration it was not given. Six transforms in one
process at widths of three, eight, three, the default, eight again and three
come back at 2.0e-03, 4.0e-06, 2.0e-03, 4.6e-04, 1.4e-06 and 2.0e-03: every
repeat is the same number to every digit.

On a 64-grid of 64 spokes, what each kernel keeps and what compressing costs:

| sigma / os | ns | mask width | kept, this | kept, BART | cost, this | cost, BART |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | 2 | 84% | 82% | 2.4e-02 | -- |
| 2 | 6 | 3 | 86% | 84% | 1.4e-02 | 2.5e-02 |
| 2 | 8 | 4 | 88% | 86% | 9.6e-03 | -- |
| 1.25 | 4 | 4 | 88% | -- | 1.1e-02 | -- |
| 1.25 | 6 | 5 | 88% | -- | 9.6e-03 | -- |
| 1.25 | 8 | 7 | 93% | -- | 4.5e-03 | -- |

Two points of grid apart at every width, which is one rounding of a width to
whole cells and not a drift; the same to within a point on a card. BART has no
column at a quarter over because it has no point spread function there, and
none at a width other than six because of its own table. A tolerance for ns=8
at an oversampling of two is below what single precision reaches, so that
kernel is the end of the range rather than a result.

`zero-mem` is the exception that is not one: it is a parenthesised flag in
BART's own help, and its Toeplitz normal does not reconstruct in BART either
-- BART's own is nearly two from BART's own default. It is intercepted like
the rest and nothing more is claimed for it.

The normal applies no roll-off, which is what makes a function computed
elsewhere fit it at all. `nufft.c` folds the roll-off into the linear phases
only `if (!conf.toeplitz)`, and `toeplitz_mult` never reads `data->roll`: it
multiplies by the phases, transforms, multiplies by the function, transforms
back. So the whole deapodisation convention sits inside the function, put
there by whatever computed it -- FINUFFT deapodises with the kernel it spread
with, that lands in the function, and the transform pair beside it is the same
library. `data->roll` survives for the forward and the adjoint, which are not
BART's here and never run.

A^H A as one convolution against A^H A as two transforms differs by 1.2e-03 at
a thousandth and 2.1e-06 at a millionth -- it closes with the tolerance, which
is what says the function is the right one rather than nearly so. A roll-off
that did not match would be a smooth error of order one across the field of
view, and would not shrink when the transform is tightened.

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
`bartorch._finufft.normals_built()` says which of the two answered.

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

**cuFINUFFT is the same table.** `src/csrc/substitute/finufft.c` holds two of them, filled
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

**`-o` is `upsampfac`, and `-w` is the tolerance read backwards.** How far
past the image the transform is computed on is the one gridding parameter both
sides spell the same way, so BART's `-o` is carried across wherever it is not
BART's own default of two; where it is, the library's own setting applies
(`_finufft.use_in_tools(upsampling=...)`, a quarter over by default).  Zero
leaves the choice to FINUFFT: a smaller grid buys a wider spreading kernel, and
which of the two costs more depends on how many samples fall on each mode. On a 160 cube with eight coils,
four coefficients and 3.5 million samples, an adjoint takes 6.7 s either way
but holds 1.29 GB rather than 1.45; at a quarter over it takes 13.2 s, because
at that density spreading is what the transform spends its time in.

A width has no field of its own -- FINUFFT sizes its kernel from the tolerance,

    ns = ceil( ln(tolfac / tol) / (pi sqrt(1 - 1/sigma)) + 1 )

with `tolfac = 0.18 * 1.4^(dim-1)` for a type 1 or 2, from FINUFFT's
`src/common/kernel.cpp` -- so `-w` is carried across by inverting it. Past
about seven grid points the kernel is no longer what limits a single-precision
transform and the tolerance it stands for falls below what one can reach, so
it is clamped there. BART's own operator cannot serve a second width in one
process at all: its Kaiser-Bessel window is built once and refuses a different
beta.

**Precision is the caller's to spend.** The default tolerance is a thousandth
(see *On a card*): that same 160 cube takes 2.1 s at a thousandth rather than
6.7 s at a tolerance an order below BART's own gridder, which is what makes it
fit on a laptop.
The setting is private (`_finufft.use_in_tools(tolerance=...)`); operators take
`oversampling` and `width`, and the tools `-o` and `-w`.  cuFINUFFT takes only
two, a quarter over, or the heuristic; `-o 1.5` plans on the host and fails on a
card.

`linop.NUFFT` is that transform reached without BART's tools, for
chaining and solving in Python, and it is FINUFFT's underneath like everything
else. A BART trajectory always carries three components, so whether a
transform is two- or three-dimensional is decided by whether kz is used, not by
the trajectory's shape -- which is also what says how many of an image's
trailing axes are spatial and how many are coils.

**A tool keeps the card where that is safe; an operator always does.** Two
things stand between a BART tool and the memory it was handed. The few entry
points that read an array element by element rather than through `md_` --
`estimate_im_dims` sizing an image from a trajectory, `estimate_scaling_norm`
taking a median of k-space -- are answered in `src/csrc/abi/host_reads.c` over a host
copy of that one array, which is what BART already does for its own virtual
pointers. What cannot be reached that way is a tool that allocates a temporary
of its own on the host and mixes it with its input: `pocsense` takes its
pattern from `md_alloc`, `nlinv` from `anon_cfl`, `ecalib` sets `bart_use_gpu`
from its own flag. `md_` operations take the host path unless every argument
is on a device, and take it silently, so that is a segmentation fault rather
than a slower answer.

Which tools are which is a property of their own code, so `_ON_DEVICE` in
`_dispatch.py` holds only what has been run on a card and checked against the
host, and a test in `tests/test_cuda.py` runs every name in it. The rest are
given host memory and their result comes back on the card. On a 256x256
eight-coil radial dataset that is `pics` in 0.12 s rather than 0.25 s and
`fft` in 3 ms rather than 10 ms.

Whichever way a tool runs, what BART allocates for itself comes from torch on
the memory that tool was given: an output on the other side of the bus from
its input is the same silent host path.  It comes zeroed, as a new CFL file
does: `rof` and `tgv` start their solver from what the output already holds. BART also maps input files
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

Windows is not a platform here. BART does not build on it, and nothing in
this repository carries a path toward one: no `.dll` among the names the
loader tries, no `__declspec(dllexport)`, no `win32` branch picking a
different library. WSL2 is a Linux install and is the answer.

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
`src/csrc/abi/api.c` answers `__assert_fail` instead, which needs neither the define
nor the libraries, and the symbol is hidden so it binds inside this library
alone.

## The operator layer

`LinearOperator` is the one operator class: two shapes, a forward, an adjoint,
a normal.  A subclass is defined either by `_create`, which builds one of
BART's operators -- `FFT`, `Diagonal`, `Sampling`, `MultiplySum`, `NUFFT`,
`NoncartesianSense`, `Callback`, `Compose`, `Add` -- or in Python by `forward` and
`adjoint`.  A BART-backed subclass says only which constructor makes it; the
lock BART is called under, the device it is built on, the handle's lifetime and
the tensors the handle holds by pointer are all in `_operator.py`, shared with
`NonlinearOperator`, which is built the same way.

Composition builds BART's composite rather than a Python chain, so a chain of
five applies as one call and a solver iterating on it never returns to Python.
A Python-defined operator joins through `_bart()`, which wraps its methods as a
`Callback` -- a new one each time, so the operator holds no handle that refers
back to itself.

`.H` is the exception that proves the rule: BART has no adjoint-of-an-operator
constructor, so `Adjoint` is the operator read the other way round rather than
a second handle, and applying it costs what `adjoint` costs. Only composing it
needs a handle, and only then is one made.

`NoncartesianSense` takes `coil_batch` and `fold_maps`, and each is a field of
the form the operator is built from, so a build reads nothing from process-wide
state.  The process-wide values (`_dispatch.set_coil_batch`,
`_dispatch.set_fold_maps`) are what BART's own tools read when they build a
SENSE operator, and nothing else.

## The encoding form

Every MRI encoding reduces to one expression, for coil `c`, encoding frame `t`
and sample `k`:

```
y[c, t, k] = sum_a  O[a, t](k) . T_t( I[c, a, t](r) . x[a](r) )(k)
```

`I` is the image-side element-wise factor -- the coil sensitivities, and a
contraction's spatial weights; `T` is the transform, one of an FFT on the
image's grid, a NUFFT over a trajectory, and a wave; `O` is the k-space
element-wise factor -- a pattern, a table of sampled phase encodes, a subspace
basis, density weights, a contraction's sample weights; and `sum_a` is the
contraction.  `linop/form.py` is that expression as the library takes it, and
`struct bartorch_encoding` is the same record in C.

There is one executor, and it is parameterised by the form rather than written
once per encoding.  `bartorch_linop_encoding` is the only entry point that
builds an MRI encoding: `src/csrc/ops/sense.c` decides whether the coils can be
walked a slab at a time, holds the bank as maps or inflates it from kernels,
puts the contraction's terms around the transform, and asks `grid.c` or
`nufft_create2` for the transform itself.  What the four encodings differ by is
one `switch`.

**Composing in Python builds a description.**  Matching and lowering happen
once, when the operator is built, and each application is then one call into
the library.  `linop/plan.py` holds both halves: `describe` reads a composition
of diagonals around an encoding back into chains and sums of them, `lower`
folds the factors into the encoding's form, and `materialise` builds BART's
plain sum of chains where they do not fit.

So `@` and `+` build nothing.  A composition works its shapes out from its
operands and defers, and the first thing that needs the handle -- an
application, `.H`, a solve, reading `.plan` -- is what offers the whole
description to the planner.  Lowering a product as it is written would see one
factor at a time and a sum of lowered terms is no longer a sum of chains, so
the deferral is what lets a caller write the terms themselves and get the
encoding a fit's coefficients would have given them.  `materialise` builds with
matching off, because it is the description's fallback.

What is a composition rather than an operator: off-resonance by time
segmentation, a phase per shot with the samples each shot took, and an echo
phase with a subspace basis.  Each is `sum_l diag(b_l) E diag(c_l)` over the
terms, written with `@` and `+`, and each lowers into one encoding whose
contraction is those terms -- one transform per term inside the coil loop,
which is what a per-frame image factor costs and all it costs.
`tests/test_plan.py` holds each against its sum written out with torch's own
transform, and against the plan it was meant to take.

Simultaneous multislice is the fourth, and it is the one whose sum does not
fit inside the sensitivities.  Each slice takes its own phase in k-space and
the slices add up after it, so the sets have to survive the multiply that
usually contracts them: with a slice phase in the form the coil images keep
the sets, the transform runs once per slice, and `linop_sum_create` adds them
up past the k-space factor.  The terms say which slice they are by picking it
whole on the image side, and `plan.contraction` is `slices`.

**A batch the sensitivities vary along is inside the operator.**  Independent
slices, each with their own maps over one trajectory, are
`(nz, 1, c, y, x)` -- a batch axis in front of a sets axis, which is written
even where there is one set, so that `(3, c, y, x)` is three sets and
`(3, 1, c, y, x)` three items.  The torch layout puts such an axis above the
coils, and `bartorch_linop_blocks` applies one operator to every block, so
this one cannot be a block: it is a dimension of the operator instead, the
slowest BART has (`_layout.SENS_BATCH`), which costs the encoding axes one
free dimension.  `sense_output_from` places the coils above every axis of a
block that is *not* one of these, and the samples are copied by the whole's
own strides rather than a block's, so an axis slower than the coils lands
where it belongs.

Off a grid the trajectory is shared, so the batch is one of the axes it does
not index -- which is what FINUFFT's `ntrans` is for, a batch of transforms
against one point set.  It costs one plan rather than one per item.  The
trajectory's own dimension vector therefore does *not* carry the batch, while
the image's and the samples' do: saying the trajectory varies across it would
make it a sample axis the image varies along too, which is a transform per
frame and declined.

On a grid, under the wave front and off the grid alike, each item answers
exactly what its own operator would.

Picking a slice whole is the only image factor the sets can carry.  An image
factor is applied to the coil images, where `md_ztenmul2` has already
contracted the sets, so a weight that differs between them has nowhere to go
and is left to the sum of the terms with `plan.contraction` saying `chained`.
Fusing it would be fusing a wrong answer, which is the one outcome worse than
a slow one, and a test holds it chained.

**The chosen plan is never silent.**  A fallback answers with the same numbers
several times slower, so it is reported rather than left to a timing.  `A.plan`
names the transform, the factors on each side, the contraction, what is
streamed, how the normal is applied, and which executor ran it --
`plan.executor` read back from `bartorch_encoding_counter` after the build, not
worked out a second time in Python.  `plan.fused` is false for a form the slab
loop could not take and for a sum of terms left chained.  The counters also
record the loop's applications, which is what a card test asserts.

**A backward pass is the adjoint, not the transpose.** Torch stores conjugate
Wirtinger gradients: what it wants back for `y = A x` is `A^H g`. The near
miss is silent and a real-valued test would never see it, so
`tests/test_linop.py` compares against torch's own gradient for a
multiplication torch can do, and asserts that the transpose would have
disagreed.

**`deepinv` is an adapter, not a base class.** `bartorch.interop.to_deepinv(A)`
returns a real `LinearPhysics`, whose class is built the first time it is asked
for, so deepinv is an optional extra and nothing else imports it. Inheriting
instead would put that import in the path of every operator and tie releases
here to releases there. The wrapper's own work is `deepinv`'s batch axis,
which a BART operator does not have, and `A_dagger` as `optim.CG`.

## Nothing here is an algorithm

The rule the whole package is under: everything is a wrapper around BART,
except the substitutions that exist to be faster than it -- FINUFFT and
cuFINUFFT under `nufft_create`, the coil-slab SENSE operator, the normal
operators beside them. Anything else written here would be a second
implementation that drifts, and a result that is nearly BART's is worth less
than no result.

Signal simulation is the exception in the other direction: it is TorchSim's,
not BART's.  `bloch`, `epg`, `sim`, `signal`, `mobasig`, `pulse` and `seq` are
private, because what a command returns is a curve and what a fit needs is a
model -- a forward it can differentiate, with bounds and a starting state --
which is what `nlop`'s `SignalModel` is over a TorchSim simulator.  The
commands stay reachable through `_call.build` so a test can pin TorchSim's
physics against BART's closed forms, which `tests/test_nlop_torchsim.py` does.

So `priors/` computes nothing, and the iterations write out only their
steps: the proximal steps, one block each in `optim/blocks.py`, and the
Gauss-Newton step, `IRGNMBlock` in `nlop/irgnm.py`, whose operators -- the
model, its adjoint derivative and `norm_inv`'s inverse -- are BART's.  Each is
held to the library's bits (`tests/test_optim_iterators.py`,
`tests/test_nlop_irgnm.py`) and looped by its solver.  Conjugate gradients and
NIHT go to `bartorch_solve`, which configures BART's iteration exactly as
`pics` does. A term fills
the table `opt_reg_configure` reads -- which kind, over which axes, with what
weight -- from an object rather than from a `-R` string, and holds the
proximal operator and the transform BART makes of it. The solver is handed
those, not a description to rebuild from, so solving twice with a term builds
nothing the second time. The letters are BART's own, and a test holds every
one this package offers against `grecon/optreg.c`, because a term BART does
not know is answered with `error()` and that leaves the library spinning.

Three of BART's terms cannot be built alone: TGV and the two infimal
convolutions extend the optimisation variable, and what they add is counted
across the whole set (`ropts->svars`, and the assertion at the end of
`opt_reg_configure`).  `bartorch_solve` configures the set itself when one is
present, so `tools.pics` takes them.  `optim.ADMMBlock` and `optim.PRIDUBlock`
ask `bartorch_prox_set_create` for the same set and walk the image followed by
the unknowns, with the encoding chained onto an extract of its front as
`pics.c` chains it.

**Two habits of BART's arithmetic** decide whether a loop written out in
`optim/blocks.py` answers with the library's bits, and both are easy to
undo by accident.  Every scalar is a C `float` unless the library declares a
`double`, and a scalar worked out in a double and rounded once at the end is a
different number.  And a vector is scaled by a coefficient rather than divided
by its reciprocal: `chambolle_pock` works out `1 / sigma`, `1 / (1 + sigma)`
and `-sigma / (1 + sigma)` once, in a double, and rounds each to a float;
dividing the tensor instead agrees while `sigma` is where `pics` starts it and
stops agreeing once the adaptive step has moved it.  `_single`, `_norm`,
`_ravine` and `_formula` are where that is kept.

**The order costs nothing.** A C-order tensor of shape `(a, b, c)` and a BART
array of dims `[c, b, a]` are the same bytes, so the boundary reverses the
dimension vector and hands over `data_ptr()`; an operator never copies. The
one copy is defensive and only for tools: BART maps its inputs
copy-on-write and some tools write into them, so `dispatch` clones unless
`set_copy_inputs(False)` says not to.

 `pics` turns its arguments into three things
and hands them to `lsqr2`: the proximal operators its `-R` strings name, the
algorithm its solver flag chooses, and the encoding. `src/csrc/ops/iter.c` does the
same with the same functions, in the same order, over an operator assembled
here. The loop runs where `pics`'s does, and an operator BART built is handed
over as it stands rather than wrapped, so there is no crossing per step --
`tests/test_solve.py` asserts both: one call into the library however many
iterations it runs, and `_bart()` returning the operator itself.

**And it is exact end to end.** `pics` does work around its solve, and an
assembled reconstruction does the same work in Python: the sampling pattern
applied to the k-space, `ifftmod` on it, and the scaling `pics` estimates
unless `-w` says otherwise, which is `optim.data_scaling`. What is left is what
`italgo_config` is handed, and three of those the tool fills in rather than
BART: the 0.95 step of its proximal-gradient iterations, the random cycle
spinning it does unless `-n`, and PRIDU's `sigma_tau_ratio`.
`tests/test_solve.py` holds nine configurations of `pics` against the
assembly with `torch.equal`, and they are equal.

One thing is not bit-exact and cannot be: `-e` estimates the largest
eigenvalue by a power iteration whose starting vector comes from BART's
process-global generator, so a solve here and a `pics` call in the same
process start it from different places. They agree to the power method's own
convergence. A fresh `bart` process starts that generator from a constant,
which is why the tool agrees with itself across runs and not across calls.


```sh
./scripts/run_tests.sh              # build into build/local, regenerate, run it all
./scripts/run_tests.sh tests/test_solve.py -k pics    # anything else goes to pytest
./scripts/run_tests.sh --rebuild    # after a source file moves; CMake caches the list
./scripts/build_docs.sh             # the reference, warnings as errors
pip install -e .                    # the same build through scikit-build-core
pip install -e . --config-settings=cmake.define.BARTORCH_CUDA=ON   # with device code
python scripts/check_device.py      # everything a card can answer that a host cannot
python scripts/gen_catalogue.py     # after a submodule bump
python scripts/gen_abi.py           # after changing the C header
./scripts/lint.sh                   # what the lint workflow runs
./scripts/lint.sh --fix             # format, and apply what ruff can fix itself
```

What `run_tests.sh` does by hand, for when it is in the way:

```sh
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release && cmake --build build/local -j
BARTORCH_LIBRARY=$PWD/build/local/libbartorch.so PYTHONPATH=src pytest tests/
```

Three things a fresh checkout needs before that first line works, each of
which fails with a message that does not say which:

* **A compiler that puts BART's nested functions on the heap.** GCC 14 or
  newer, for `-ftrampoline-impl=heap`, or clang -- `cmake -DCMAKE_C_COMPILER=clang
  -DCMAKE_CXX_COMPILER=clang++`. CMake says so and stops; GCC 13 is not enough.
* **OpenMP for whichever of those it is.** `libomp-dev` beside clang. Without
  it the build still works and the overlapped walks in `src/csrc/ops/sense.c` run in
  sequence.
* **FINUFFT**, which `pip install -e .` brings on every platform it ships a
  wheel for. Working from a source checkout on `PYTHONPATH` instead, install
  it by hand: without it seventeen tests fail rather than skip, because
  nothing reaches BART's own gridder without having been sent there and the
  substitution declining is an error, not a fallback.

With those, and `pip install torch numpy scipy pytest`, the suite is green
apart from the CUDA tests, which skip without a card. `pip install mkl
deepinv` covers the rest: the FFT tests that assert MKL planned, and the
``LinearPhysics`` adapter.

## Tests

A numerical test pins BART against something outside BART: numpy's FFT, an
explicit discrete Fourier sum, a closed-form signal, an adjoint identity. A
test that compares BART to BART proves nothing.

## On a card

`scripts/check_device.py` walks the device path in dependency order, each
check independent and naming its own reason, so the first failure is the thing
to fix. On an RTX 4060 Laptop, CUDA 12.8, all nine pass: `fft` against numpy
to 1e-07, the NUFFT against an explicit discrete Fourier sum to 6.7e-04 on the
host and 6.9e-04 on the card, the two agreeing with each other to 7.5e-04, and
`-g` changing nothing that the device pointers had not already decided. Those
three are the tolerance the plans are made with, not a floor anybody hit: a
thousandth by default, and the checks are held to a small multiple of whatever
is in force rather than to a number of their own.

On a 256x256 eight-coil radial dataset of 401 spokes, best of five, `pics`
takes 0.17 to 0.25 s on the card with the point spread function and 0.11 to
0.28 s with the transform pair -- each inside the other's scatter -- against
2.2 to 2.4 s and 3.1 to 3.8 s on the same machine's host. The pair is cheap
enough on a card that halving the number of transforms stops being worth
measuring; on the host the point spread function is still worth about half the
time. This is a 45 W laptop card, so a run measured cold and one measured
after the clocks have dropped differ by more than the two normals do.

More than one BART stream makes no difference that measurement can resolve:
one, two and four streams reconstruct in the same sixth of a second, and the
spread between them is smaller than the spread between repetitions of any one
of them. What BART holds on the card is what `bartorch.use_cuda_memcache`
decides for an operator -- 34 MB against nothing on that dataset -- and
nothing at all for a tool, because BART's `main` clears the cache when a
command ends.

The default transform is a thousandth on a grid a quarter over, because a
reconstruction is held to its data rather than to its transform. Two
exceptions: a caller who names an upsampling gets it -- `nufft_conf_s` carries
two as BART's own default, so zero is what says nobody asked, and BART gets
its two back before it sees the conf -- and the tools that calibrate before a
reconstruction is attempted (`nlinv`, `rtnlinv`, `ncalib`, in `_CALIBRATES`)
run at FINUFFT's own tolerance on the textbook grid, which costs a few
megabytes at the resolution they fit sensitivities at.

Every BART entry point that builds a NUFFT is served: `nufft` forward,
adjoint, inverse and Toeplitz, `pics` with and without a pattern, `sqpics`,
`nlinv`, `rtnlinv`, `moba`, `ncalib` and `linop.NUFFT`, on the host
and on the card, with BART's own gridder built zero times. `nlinv` and the
network models build theirs against dimensions alone and hand the trajectory
over afterwards, which is why `nufft_update_traj` installs one rather than
refusing.

## Conventions

Shapes are C order; a BART dimension vector is the reversed shape. Wherever
BART takes a bitmask or a dimension number, Python takes axis indices, and
index sets that are not axes (channels, parameter maps) are tuples too; no
public argument takes a bitmask or a `-R` string. Hand-written wrappers
convert their own; derived wrappers and what a curated one passes through by
name follow `_call.TRANSLATED`, and `test_tools` fails on a dimension-like
argument that is in neither. `pics`, `sqpics`, `wshfl` and `moba` take
`bartorch.priors` terms, serialized by `Regularizer._argument`. A flag's value can be an
array rather than a number -- `pics(kspace, maps, t=traj)`, `-p` for a
sampling pattern, `-B` for a basis -- and is registered and copied like any
other input. BART's `fft` tool is
unnormalised unless asked for the unitary form; the `linop.FFT`
operator is unitary. `nufft` output is scaled by one over the
square root of the image's voxel count, with a negative exponent.

Write for someone reading the code as it is now. No text about what the code
used to be. Docstrings follow *Documentation* below.

## What is not done

Tools with optional extra outputs.  BART's other operator constructors --
resize, transpose, sum, finite differences, wavelets, exponentials and the rest
in `linops/` and `nlops/` -- which would let an application assembled here stay
one BART operator.  `ictv`, which fails inside BART for every input
(`ictv.c:97` reshapes the wrong side of an operator).

`signal -C` is guarded rather than fixed.  `ir_multi_grad_echo_model`
(`simu/signals.c:461-467`) fills `(N / NE) * NE` entries of the `N`-entry
stack array `signal.c:234` leaves uninitialized, and `get_signal` averages all
`N` of them into the output; without `-m` at all, `NE` is BART's `-1` and none
of it is written.  What comes back is finite stack memory.  `_call`'s
`_PRECONDITIONS` refuses those combinations before the command runs, which is
where any further "BART does not define this" case belongs.

Windows is not on this list because it is not a target: BART does not build
there, and WSL2 is a Linux install like any other.

A tool that takes device memory as it stands. BART guards the host reads that
would break -- `estimate_im_dims` copies to the host when it is handed one --
for its own virtual pointers, not for a raw device pointer, so the route in is
`src/csrc/abi/memcfl.c` handing BART a `vptr_wrap_cfl` rather than the pointer itself,
and then finding out which of BART's guards are complete.

Two of mrtoeplitz's ideas have no route in from here: a transfer that stays on
the host and is streamed across in chunks, and bfloat16 transfers, which halve
what crosses the bus. Both are decisions about how the point spread function
is stored, which lives inside `nufft.c`, so neither is reachable by
substituting an entry point -- they would need a BART edit or a normal
operator written here. The seam is one function: `toeplitz_for` in
`src/csrc/substitute/nufft_finufft.c` decides what the operator's normal is, and an
mrtoeplitz kernel behind a host callback would go there. What BART does have
is `compress_psf`, `decomposed_psf` and `lowmem`, and its own overlap:
`bartorch.set_cuda_streams` sets `cuda_num_streams`, which is what puts BART's
transfers and its arithmetic on different streams.

A Cartesian trajectory axis costing no set of frequencies of its own.  BART
doubles and decomposes the axes in `conf.flags` and nothing else per axis
(`conf.decomp` is one boolean for all of them), and `flags` is also what
`nufft_create2` checks the sample count against -- so an axis taken out of it
to skip the doubling is an axis BART then measures the image by, and refuses.
`conf.cfft` has the right shape but wants the axis to be a k-space axis of the
data rather than a component of the trajectory.  So it needs the samples
lifted out of the raveled point set, or a BART edit.

An image that varies along an axis the trajectory indexes: one plan per item,
which the substitution declines.  A stacked normal kernel for per-item
trajectories would need those plans as well as the stack.

Routing BART's device allocations through torch's allocator is a further step:
`mem_device_malloc` takes the allocator as a parameter, so replacing
`num/mem.c` would do it without a BART edit.

## Documentation

`docs/guides/developer/documentation.md` is the authoritative documentation guide for both contributors and agents. Read and follow it before creating or substantially modifying documentation, docstrings, examples, tutorials, or explanatory material.

Treat its distinction between API reference, gallery examples, conceptual explanation, and tutorials/how-to guides as a requirement. Do not transfer the prose style or level of exposition of one documentation type into another.

Project-specific terminology and conventions take precedence over generic examples in the guide.

When modifying existing documentation:

* verify substantive semantics against implementation, tests, authoritative upstream specifications/libraries, and literature where appropriate;
* do not treat existing prose as authoritative;
* preserve technically good documentation and avoid unrelated stylistic churn;
* flag unresolved semantic discrepancies rather than guessing;
* when useful material is in the wrong documentation type, move or develop it in the appropriate location rather than automatically deleting it.

When auditing or refactoring documentation, explicitly check for conversational or literary LLM prose, paraphrastic replacements for established technical terminology, personification, taglines, code narration, and technically imprecise attempts at accessibility.

After substantial documentation work, build the documentation, run relevant documentation tests/examples, and inspect the rendered output.

The documentation is in four parts, and a page belongs to whichever it is:
`docs/api/` is the reference, extracted from the docstrings; `docs/examples/`
is the example gallery, executable scripts rendered by sphinx-gallery;
`docs/explanation/` is the conceptual material; `docs/guides/` is the user and
developer guides.

An example is a Python script under `docs/examples/<section>/`, named
`NN-title.py`, whose module docstring becomes the page and whose numeric prefix
orders it within its section. A section is a directory with a `README.rst`
holding its heading and a paragraph; `docs/conf.py` lists the sections in the
order a reader meets them. Code that is not about this library -- figure
layout, colormaps, the phantom's arithmetic -- goes between
`# sphinx_gallery_start_ignore` and `# sphinx_gallery_end_ignore`, which keeps
it off the page and in the downloadable script and notebook. Anything a reader
would type themselves stays visible.

`./scripts/build_docs.sh` renders the example pages without running them, which
needs no compiled library; `--execute` runs them, which needs one and the
packages `docs/examples/README.rst` names. The docs workflow does both, and
publishes the executed build.

