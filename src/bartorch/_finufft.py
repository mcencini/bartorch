"""FINUFFT, as an operator and underneath BART's own tools.

The ``finufft`` and ``cufinufft`` wheels carry a compiled library and a thin
Python wrapper over it, so a plan made here is the same object BART would
have made for itself, and the transform runs with the GIL released.  A plan is
made once per operator and reused, which is what makes this worth doing inside
an iterative solve.

The scaling and sign follow BART's own NUFFT, so an operator from here is
interchangeable with :class:`bartorch.linop.NUFFT`: a type-2
transform with a negative exponent, divided by the square root of the number
of voxels.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
from pathlib import Path

import torch

_keepalive: list[object] = []

Shape = tuple[int, ...]


def available() -> bool:
    """Whether the ``finufft`` package is installed."""
    try:
        import finufft  # noqa: F401
    except ImportError:
        return False
    return True


def required_but_missing() -> str:
    """Why there is no FINUFFT here, given that there should always be one.

    It is a dependency rather than an extra, so its absence is a broken
    install and never a choice -- which is worth saying, because the obvious
    reading of "FINUFFT is not in use" is that some option was left off.
    """
    return (
        "FINUFFT computes every non-Cartesian transform here: BART's own "
        "gridder is not reachable from this package's surface.  It is a "
        "dependency of bartorch rather than an extra, so it should already be "
        "installed and something has removed it.  pip install finufft"
    )


def cuda_available() -> bool:
    """Whether the ``cufinufft`` package is installed."""
    try:
        import cufinufft  # noqa: F401
    except ImportError:
        return False
    return True


def used_on_device() -> bool:
    """Whether a transform BART runs on a device would be computed by cuFINUFFT."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_usable_on(1))


def _library_path(package: str, stem: str) -> str | None:
    """The compiled library inside a FINUFFT wheel."""
    try:
        module = importlib.import_module(package)
    except ImportError:
        return None
    here = Path(module.__file__).resolve().parent
    for name in (f"lib{stem}.so", f"lib{stem}.dylib", f"{stem}.dll", f"lib{stem}.dll"):
        candidate = here / name
        if candidate.exists():
            return str(candidate)
    return None


#: What an OpenMP runtime's library is called, whoever built it.
_OPENMP_LIBRARIES = ("libomp.", "libiomp5.", "libgomp.")


def openmp_runtimes() -> list[str]:
    """The OpenMP runtimes loaded into this process, by path.

    Only macOS answers: LLVM's runtime refuses to initialise where another
    copy already has, and refuses by calling ``abort()`` -- so two of them in
    one process is not a warning to weigh but the end of the interpreter,
    with the reason on a stderr that whoever is capturing output never shows.
    Linux is not asked because glibc's loader resolves the duplicate instead
    of dying on it.

    Returns
    -------
    list of str
        One path per loaded runtime, in load order.  Empty off macOS.
    """
    import ctypes as c

    if sys.platform != "darwin":
        return []

    try:
        dyld = c.CDLL(None)
        dyld._dyld_image_count.restype = c.c_uint32
        dyld._dyld_image_count.argtypes = []
        dyld._dyld_get_image_name.restype = c.c_char_p
        dyld._dyld_get_image_name.argtypes = [c.c_uint32]
        loaded = [dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count())]
    except (AttributeError, OSError):  # pragma: no cover - macOS only
        return []

    found = []
    for name in loaded:
        if name is None:
            continue
        path = name.decode(errors="replace")
        base = path.rsplit("/", 1)[-1]
        if base.startswith(_OPENMP_LIBRARIES):
            found.append(path)
    return found


def _load_symbols() -> bool:
    """Hand the library FINUFFT's entry points and its options layout.

    The host's library is required; the device's is loaded when the
    ``cufinufft`` wheel is installed, and its absence only means that a
    transform BART would run on a card stays with BART's own operator.

    A layout is read from the package that will interpret the struct, so a
    release that moves a field cannot be misread here.
    """
    return _load_one(0, "finufft", "finufft", "finufftf_") and (
        _load_one(1, "cufinufft", "cufinufft", "cufinufftf_") or True
    )


def _load_one(device: int, package: str, stem: str, prefix: str) -> bool:
    """Register one library's entry points and the offset of the field to set."""
    import ctypes as c

    from bartorch._lib import library

    lib = library()
    path = _library_path(package, stem)
    if path is None:
        return False

    try:
        opts, field, upsampling, spreadonly = _options_layout(package)
    except (ImportError, AttributeError):
        return False

    handle = c.CDLL(path)
    _keepalive.append(handle)
    symbols = [prefix + name for name in ("makeplan", "setpts", "execute", "destroy")]
    symbols.append(_default_opts_symbol(handle, prefix))
    for symbol in symbols:
        fn = getattr(handle, symbol, None)
        if fn is None:
            return False
        if lib.bartorch_finufft_set(symbol.encode(), c.cast(fn, c.c_void_p)) != 0:
            return False

    return 0 == lib.bartorch_finufft_layout(
        device, c.sizeof(opts), field.offset, upsampling.offset, spreadonly.offset
    )


def _default_opts_symbol(handle, prefix: str) -> str:
    """FINUFFT spells its defaults per precision and cuFINUFFT does not."""
    name = prefix + "default_opts"
    if getattr(handle, name, None) is not None:
        return name
    return prefix.replace("f_", "_") + "default_opts"


def _options_layout(package: str):
    """The options struct a package interprets, and the fields to fill in.

    FINUFFT is told how many threads to take -- zero, meaning all of them --
    and cuFINUFFT which device to run on; both are told how far past the image
    to spread, which they spell alike, and whether to spread and stop there,
    which they do not.
    """
    if package == "finufft":
        from finufft._finufft import FinufftOpts as opts

        return opts, opts.nthreads, opts.upsampfac, opts.spreadinterponly
    from cufinufft._cufinufft import NufftOpts as opts

    return opts, opts.gpu_device_id, opts.upsampfac, opts.gpu_spreadinterponly


def use_in_tools(
    enable: bool = True,
    tolerance: float = 1e-3,
    upsampling: float = 1.25,
) -> bool:
    """Have BART's own tools compute their NUFFT with FINUFFT.

    Parameters
    ----------
    enable : bool
        Turn the substitution on, or off to leave BART its own gridder.
    tolerance : float
        The tolerance FINUFFT plans are made with.  A thousandth by default:
        a reconstruction is not made better by a transform an order more
        accurate than the data going into it, and the kernel narrows as the
        tolerance loosens.
    upsampling : float
        How far past the image to spread before transforming.  Two is the
        textbook grid; a quarter over, the default here, trades a smaller one
        for a wider kernel and holds a quarter of the memory at this
        tolerance.  Zero leaves the choice to FINUFFT, per problem.  BART's
        ``-o`` takes precedence wherever it is not BART's own default of two.

    Returns
    -------
    bool
        Whether the substitution is in place and agrees with BART.

    Raises
    ------
    ImportError
        When ``finufft`` is missing, or ``cufinufft`` is missing on a machine
        where BART would otherwise run on a card.
    """
    from bartorch import _cuda
    from bartorch._lib import library

    lib = library()
    if not enable:
        lib.bartorch_finufft_use_in_tools(0)
        lib.bartorch_nufft_allow_fallback(1)
        return False

    if not available():
        raise ImportError(required_but_missing())

    if _cuda.available() and not cuda_available():
        raise ImportError(
            "this machine has a device BART can use, and cuFINUFFT is what would serve it: "
            "pip install 'bartorch[cufinufft]'"
        )

    if not _load_symbols():
        raise ImportError(
            "the finufft package is installed but its library did not hand over the entry "
            "points this needs; check that it matches the version pyproject.toml asks for"
        )

    # Loading FINUFFT's library is safe; calling into it is what starts its
    # OpenMP runtime, and starting a second one is what LLVM's answers with
    # abort().  torch brings one and the macOS FINUFFT wheel brings its own,
    # so on that platform the pair is checked before the first call rather
    # than found out by the process ending.  Continuing anyway is what
    # KMP_DUPLICATE_LIB_OK asks for, and what it buys is a crash later or a
    # wrong answer quietly -- neither of which a reconstruction should risk.
    runtimes = openmp_runtimes()
    if len(runtimes) > 1:
        lib.bartorch_finufft_use_in_tools(0)
        lib.bartorch_nufft_allow_fallback(1)
        raise RuntimeError(
            "this process has loaded more than one OpenMP runtime ("
            + ", ".join(runtimes)
            + "), and calling FINUFFT would start the second, which LLVM's runtime ends "
            "the process over (OMP: Error #15).  BART's own gridder serves the transforms "
            "instead.  One runtime is the fix: a FINUFFT built against the same libomp "
            "torch carries, or a torch built against FINUFFT's"
        )

    lib.bartorch_finufft_set_tolerance(float(tolerance))
    lib.bartorch_finufft_set_upsampling(float(upsampling))
    lib.bartorch_nufft_allow_fallback(0)
    lib.bartorch_finufft_use_in_tools(1)

    if not _tools_agree_with_bart():
        lib.bartorch_finufft_use_in_tools(0)
        raise RuntimeError(
            "FINUFFT is installed but its NUFFT does not agree with BART's own; "
            "the substitution has been left off"
        )

    return bool(lib.bartorch_finufft_usable())


def used_in_tools() -> bool:
    """Whether BART's tools are computing their NUFFT with FINUFFT."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_usable())


def stream_psf(enable: bool = True) -> None:
    """Keep the function a Toeplitz normal convolves with off the card.

    BART reads it as one array and takes the set of frequencies it wants out
    of it, so it brings the whole of it over the first time a normal is
    applied -- and for a subspace problem that function is coefficients by
    sets by image, which is what a three-dimensional reconstruction cannot
    fit.  Asked for this, the loop over sets is driven here instead: BART is
    left believing it has one, and the one it has is brought over in turn.

    It costs BART's low-memory normal, which walks the sets rather than
    convolving them at once.  On a 96^3 problem with eight coils that is
    364 MB and 2.0 s against 210 MB and 2.4 s -- two fifths of the memory for
    a fifth more time.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_stream_psf(int(bool(enable)))


def streaming_psf() -> bool:
    """Whether the function is being kept off the card."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_stream_psf())


def compress_psf(enable: bool = True) -> None:
    """Keep only the places the samples reach of the function.

    A compressed function is the part of the function the samples reached,
    which is also all that crosses the bus for every set of frequencies it is
    brought over in.  It costs an index over the grid -- one ``long`` a point
    -- so it is kept only where it gives back more than that, which is decided
    when the function is built: a subspace function, a triangle of volumes a
    set, over a trajectory that leaves enough of the grid unreached.  A scalar
    function never is.  :func:`functions_compressed` says when it happened.

    The function is not zero where the samples do not reach, only small, so
    compression is not exact.  A three-dimensional radial readout that reaches
    the edge leaves the corners of the cube, and there one normal at 64^3 over
    four coefficients is 4.8e-03 from the pair of transforms it stands for,
    against 4.3e-03 for the whole function.  A readout covering half of a
    two-dimensional grid leaves most of it: 1.9e-02 against 3.5e-03.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_compress_psf(int(bool(enable)))


def compressing_psf() -> bool:
    """Whether only the places the samples reach are kept."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_compress_psf())


def overlap_psf(enable: bool = True) -> None:
    """Bring a set of frequencies over while the one before it is convolved.

    The crossing goes on a stream of its own, ordered against BART's by events
    rather than by a second host thread, which would make BART's own threading
    nested.  The function is page-locked on the host, because an asynchronous
    copy out of pageable memory is not one.

    It costs a second slot on the card, and a slot is one set of the function:
    at 256^3 over four coefficients, 354 MiB on a normal that otherwise peaks
    at 2.4 GiB.  What it buys is the crossing, which is a twentieth of a
    normal there: 6.3 s against 6.6 s.  Off unless asked for.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_overlap_psf(int(bool(enable)))


def overlapping_psf() -> bool:
    """Whether a set crosses while the one before it is convolved."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_overlap_psf())


def _contraction_kernel(enable: bool = True) -> None:
    """Whether a real upper-triangular contraction runs in bartorch's kernel or BART's.

    BART's is kept to be held against: the two compute the same thing.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_contraction_kernel(int(bool(enable)))


def release_transforms(enable: bool = True) -> None:
    """Let the device's transform pair go at the first normal.

    With a Toeplitz function built, a normal is a convolution and reads neither
    the FINUFFT plans nor the sample positions they were set on.  A solve forms
    its right-hand side with one adjoint and then applies only normals, so the
    plans would otherwise stay on the card for every iteration with nothing
    reading them -- and a plan grows with the number of samples, which is what
    a many-frame acquisition has most of.  The first normal lets them go; a
    transform asked for afterwards plans again from the trajectory the host
    keeps.  On unless turned off.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_release_transforms(int(bool(enable)))


def releasing_transforms() -> bool:
    """Whether the device's transform pair is let go at the first normal."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_release_transforms())


def fft_callbacks(enable: bool = True) -> None:
    """Run the passes around each volume's transform inside the transform.

    A streamed, compressed set is convolved a volume at a time, and each
    volume's forward and inverse transform come with passes of their own: the
    set's phase and the coil's sensitivity on the way in and the gather after
    it, the scatter before the inverse and the conjugates and the sum after
    it.  cuFFT links callbacks into its kernels, so each pass becomes part of a
    read or a write the transform makes anyway.  Linking them in takes cuFFT's
    LTO callbacks and nvJitLink beside it; where either is missing the passes
    run on their own.  On unless turned off.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_fft_callbacks(int(bool(enable)))


def pair_sets(enable: bool = True) -> None:
    """Convolve the sets of a Toeplitz function in pairs, where the pair kernels allow it.

    The eight sets of a decomposed function differ by half a cell along each
    axis, and the phase along one axis passes through the transforms along the
    other two, so the two sets that differ only along x share their transforms
    along z and y, forward and back.  The pair kernels (cuFFTDx, compiled for
    the grid sizes the library was built with) convolve a coil against both
    at once.  They take a compressed real function kept as its upper triangle,
    four coefficients and a cubic grid of a compiled size; anything else is
    convolved a set at a time.  Read when a function is streamed, so it applies
    to operators built afterwards.  On unless turned off.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_paired(int(bool(enable)))


def pairing_sets() -> bool:
    """Whether the sets of a Toeplitz function are convolved in pairs where they can be."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_paired())


def bfloat16_function(enable: bool = True) -> None:
    """Keep a Toeplitz function whose sets are paired in bfloat16.

    Half the host copy, half of what crosses to the card and of the slot it
    lands in, and half of what the pass along x reads, at a rounding of 2^-9 of
    each value where float32 keeps 2^-24.  bfloat16 keeps float32's exponent,
    so no value is clipped.  Read when a function is streamed, so it applies to
    operators built afterwards.  On unless turned off.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_bf16(int(bool(enable)))


def storing_bfloat16() -> bool:
    """Whether a Toeplitz function whose sets are paired is kept in bfloat16."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_bf16())


def paired_built() -> bool:
    """Whether the library was built with the pair kernels (``BARTORCH_MATHDX_DIR``)."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_paired_built())


def functions_bfloat16() -> int:
    """Toeplitz functions kept in bfloat16 since the counters were reset."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(6))


def using_fft_callbacks() -> bool:
    """Whether the passes around a volume's transform run inside it where they can."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_fft_callbacks())


def live_plans() -> int:
    """FINUFFT plans made and not yet destroyed.

    A plan belongs to whatever asked for one -- a transform operator, a point
    spread function, the spreading a compressed one is masked with -- and
    outlives none of them, so this is back at zero once the last of them has
    been freed.
    """
    from bartorch._lib import library

    return int(library().bartorch_finufft_live_plans())


def decline_reason() -> str:
    """Why the last operator was BART's rather than FINUFFT's; empty if it was FINUFFT's."""
    from bartorch._lib import library

    return library().bartorch_nufft_decline_text().decode()


def operators_built() -> tuple[int, int]:
    """NUFFT operators built since the last reset, by FINUFFT and by BART."""
    from bartorch._lib import library

    lib = library()
    return int(lib.bartorch_nufft_counter(0)), int(lib.bartorch_nufft_counter(1))


def normals_built() -> tuple[int, int]:
    """Normal operators since the last reset: by a point spread function, by the pair.

    A^H A is a convolution, so BART answers it with one multiply against a
    point spread function rather than a forward and an adjoint transform, and
    the substituted operator borrows that for its normal.  ``pics
    --no-toeplitz`` and ``nufft -t`` are what decide whether there is one.
    """
    from bartorch._lib import library

    lib = library()
    return int(lib.bartorch_toeplitz_counter(0)), int(lib.bartorch_toeplitz_counter(1))


def functions_compressed() -> int:
    """Toeplitz functions built compressed since the counters were reset.

    Whether a function is compressed is decided when it is built, from how much
    of the grid the samples reach, so the arguments alone do not say -- this is
    how a caller or a test finds out.
    """
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(2))


def functions_real() -> int:
    """Toeplitz functions stored as floats since the counters were reset.

    Whether a function is stored real is decided from the basis when it is
    built, so the arguments alone do not say -- this is how a caller or a test
    finds out.
    """
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(4))


def pairs_convolved() -> int:
    """Pairs of sets convolved together by the pair kernels since the counters were reset."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(5))


def sets_through_callbacks() -> int:
    """Sets convolved with the passes inside cuFFT's transforms since the counters were reset."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(3))


def reset_counters() -> None:
    """Start counting operators and normal operators again."""
    from bartorch._lib import library

    lib = library()
    lib.bartorch_nufft_reset_counters()
    lib.bartorch_toeplitz_reset_counters()


def tolerance() -> float:
    """The tolerance FINUFFT plans are made with."""
    from bartorch._lib import library

    return float(library().bartorch_finufft_tolerance())


def upsampling() -> float:
    """How far past the image FINUFFT spreads before it transforms.

    A quarter over by default.  BART's ``-o`` is the same number and takes
    precedence wherever it is not BART's own default of two; zero leaves the
    choice to FINUFFT.
    """
    from bartorch._lib import library

    return float(library().bartorch_finufft_upsampling())


def set_threads(n: int) -> None:
    """How many threads a transform on the host takes.

    Zero, the state this starts in, leaves the count to FINUFFT, which takes
    a thread per physical core.  :func:`bartorch.set_num_threads` sets this
    along with BART's own count, so one number covers the process; passing
    zero here is how that is undone without giving BART a count of its own.

    A card has no say in it: cuFINUFFT carries a device number where FINUFFT
    carries a thread count.
    """
    from bartorch._lib import library

    library().bartorch_finufft_set_threads(int(n))


def threads() -> int:
    """Threads a transform on the host takes; zero means FINUFFT chooses."""
    from bartorch._lib import library

    return int(library().bartorch_finufft_threads())


def fallback_allowed() -> bool:
    """Whether BART's own operator may answer what FINUFFT will not."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_fallback_allowed())


def _tools_agree_with_bart(tolerance: float = 1e-2) -> bool:
    """Whether BART's NUFFT tool computes the same thing either way.

    The two are held to each other rather than to a reference, so what the
    tolerance has to allow for is BART's own gridding error, not FINUFFT's.
    """
    import bartorch.tools as bt
    from bartorch._lib import library

    lib = library()
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    image = bt.phantom([n, n]).reshape(1, n, n)

    lib.bartorch_finufft_use_in_tools(1)
    fast = bt.nufft(traj, image)

    with barts_own_gridder():
        reference = bt.nufft(traj, image)

    lib.bartorch_finufft_use_in_tools(1)

    scale = reference.abs().max()
    if scale == 0:
        return False
    return bool(((fast - reference).abs().max() / scale).item() < tolerance)


def spatial_ndim(traj: torch.Tensor) -> int:
    """Whether a trajectory is two- or three-dimensional.

    A BART trajectory always carries three components; a two-dimensional one
    leaves the third at zero.
    """
    if traj.shape[-1] < 3:
        return 2
    return 3 if bool(torch.any(traj[..., 2].real != 0)) else 2


@contextlib.contextmanager
def barts_own_gridder():
    """BART's Kaiser-Bessel gridder, for as long as the block lasts.

    Not part of the package's surface.  What it is for is holding the
    substitution against the thing it replaces -- the agreement check below,
    and the tests that pin one to the other -- because a caller who reached it
    by mistake would get an answer an order further from the transform and
    several times slower, with nothing to say so.
    """
    from bartorch._lib import library

    lib = library()
    allowed = lib.bartorch_nufft_fallback_allowed()
    was_in_tools = lib.bartorch_finufft_usable()

    lib.bartorch_nufft_allow_fallback(1)
    lib.bartorch_finufft_use_in_tools(0)
    try:
        yield
    finally:
        lib.bartorch_finufft_use_in_tools(1 if was_in_tools else 0)
        lib.bartorch_nufft_allow_fallback(allowed)


_installed = False


def install_once() -> None:
    """Put the substitution in place the first time anything needs it.

    A caller who has the package should not have to ask for it, and one who
    does not should hear about it when a transform wants it rather than get a
    quieter answer from BART.  Failing to install it is not an error here,
    because most of what BART does needs no NUFFT at all -- but it is said
    once, at warning level, because a substitution that quietly did not happen
    is the hardest kind of difference to find later.
    """
    global _installed
    if _installed:
        return
    _installed = True
    if not available():
        return
    try:
        use_in_tools(True)
    except (ImportError, RuntimeError) as exc:
        logging.getLogger("bartorch.finufft").warning(
            "FINUFFT is installed but was not put in BART's place, so its transforms "
            "will be BART's own gridder: %s",
            exc,
        )
