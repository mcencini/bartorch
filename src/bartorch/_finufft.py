"""FINUFFT under BART's ``nufft_create``: loading it, and the substitution's switches and counters.

The substitution installs itself on first use (:func:`install_once`).  Tests and
``scripts/check_device.py`` read the counters.
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
    """Error text for a missing ``finufft``, which is a dependency rather than an option."""
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
    """Path of the compiled library inside an installed wheel, or None."""
    try:
        module = importlib.import_module(package)
    except ImportError:
        return None
    here = Path(module.__file__).resolve().parent
    for name in (f"lib{stem}.so", f"lib{stem}.dylib"):
        candidate = here / name
        if candidate.exists():
            return str(candidate)
    return None


#: What an OpenMP runtime's library is called, whoever built it.
_OPENMP_LIBRARIES = ("libomp.", "libiomp5.", "libgomp.")


def openmp_runtimes() -> list[str]:
    """Paths of the OpenMP runtimes loaded in this process, in load order.

    Answered on macOS only, where LLVM's runtime aborts the process when a second
    copy initializes.  On Linux the loader resolves the duplicate and this returns
    an empty list.
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
    """Register FINUFFT's entry points and options layout; False if unusable.

    cuFINUFFT's are registered too when its wheel is installed.  Each layout is read
    from the package that interprets the struct.
    """
    return _load_one(0, "finufft", "finufft", "finufftf_") and (
        _load_one(1, "cufinufft", "cufinufft", "cufinufftf_") or True
    )


def _load_one(device: int, package: str, stem: str, prefix: str) -> bool:
    """Register one wheel's entry points and options offsets for ``device`` (0 host, 1 card)."""
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
    """The options struct ``package`` interprets, and the fields set in it.

    Threads (FINUFFT) or device id (cuFINUFFT), upsampling, and spread-only; the two
    packages spell the last differently.
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
    """Install FINUFFT as the NUFFT behind BART's tools and operators.

    Parameters
    ----------
    enable : bool
        False restores BART's own gridder.
    tolerance : float
        Tolerance FINUFFT plans are made with.
    upsampling : float
        Oversampling of FINUFFT's fine grid; 0 lets FINUFFT choose per problem.
        BART's ``-o`` overrides it wherever it is not BART's default of 2.

    Returns
    -------
    bool
        Whether the substitution is in place.

    Raises
    ------
    ImportError
        ``finufft`` is missing or its library lacks the entry points, or
        ``cufinufft`` is missing on a machine where BART would run on a card.
    RuntimeError
        Two OpenMP runtimes are loaded (macOS), or FINUFFT disagrees with BART's
        gridder on a test transform.
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
    #
    # The fallback is not opened: an answer from BART's gridder, an order
    # further from the transform and several times slower, is worse than no
    # answer when nobody asked for it.  So the transforms are refused, and the
    # message says what makes them work.
    runtimes = openmp_runtimes()
    if len(runtimes) > 1:
        lib.bartorch_finufft_use_in_tools(0)
        raise RuntimeError(
            "this process has loaded more than one OpenMP runtime ("
            + ", ".join(runtimes)
            + "), and calling FINUFFT would start the second, which LLVM's runtime ends "
            "the process over (OMP: Error #15).  Every non-Cartesian transform will be "
            "refused until there is one runtime: run `python scripts/macos_openmp.py "
            "patch`, which points FINUFFT's library at the copy torch carries"
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
    """Stream the Toeplitz point spread function to the card one set of frequencies at a time.

    Uses BART's low-memory normal: less device memory, more time.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_stream_psf(int(bool(enable)))


def streaming_psf() -> bool:
    """Whether the function is being kept off the card."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_stream_psf())


def compress_psf(enable: bool = True) -> None:
    """Store a Toeplitz function only where the samples reach, beside an index of those points.

    Decided for each function when it is built: kept for a subspace function whose
    trajectory leaves enough of the grid unreached, never for a scalar function.
    The function is small but not zero elsewhere, so compression is approximate.
    :func:`functions_compressed` counts compressed functions.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_compress_psf(int(bool(enable)))


def compressing_psf() -> bool:
    """Whether only the places the samples reach are kept."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_compress_psf())


def overlap_psf(enable: bool = True) -> None:
    """Transfer the next set of frequencies to the card while the current one is convolved.

    Needs a second device slot of one set's size and a page-locked host copy.  Off
    by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_overlap_psf(int(bool(enable)))


def overlapping_psf() -> bool:
    """Whether a set crosses while the one before it is convolved."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_overlap_psf())


def _contraction_kernel(enable: bool = True) -> None:
    """Choose bartorch's or BART's kernel for the real upper-triangular contraction."""
    from bartorch._lib import library

    library().bartorch_nufft_set_contraction_kernel(int(bool(enable)))


def release_transforms(enable: bool = True) -> None:
    """Free the device's FINUFFT plans at the first normal application.

    With a Toeplitz function built, a normal reads neither the plans nor the sample
    positions; a later forward or adjoint plans again from the host trajectory.  On
    by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_release_transforms(int(bool(enable)))


def releasing_transforms() -> bool:
    """Whether the device's transform pair is let go at the first normal."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_release_transforms())


def fft_callbacks(enable: bool = True) -> None:
    """Run the per-volume phase, sensitivity, gather and scatter passes as cuFFT LTO callbacks.

    Applies to streamed, compressed sets.  Where cuFFT's LTO callbacks or nvJitLink
    are unavailable the passes run separately.  On by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_fft_callbacks(int(bool(enable)))


def pair_sets(enable: bool = True) -> None:
    """Convolve Toeplitz sets that differ only along x in pairs, sharing their z and y transforms.

    Uses the cuFFTDx pair kernels, compiled for fixed grid sizes.  Applies to a
    compressed real function kept as its upper triangle, with four coefficients,
    on a cubic grid of a compiled size; other functions are convolved a set at a
    time.  Read when a function is streamed, so it affects operators built
    afterwards.  On by default.
    """
    from bartorch._lib import library

    library().bartorch_nufft_set_paired(int(bool(enable)))


def pairing_sets() -> bool:
    """Whether the sets of a Toeplitz function are convolved in pairs where they can be."""
    from bartorch._lib import library

    return bool(library().bartorch_nufft_paired())


def bfloat16_function(enable: bool = True) -> None:
    """Store paired Toeplitz functions in bfloat16.

    Halves the host copy, the transfer and the device slot.  Values are rounded to
    a relative 2^-9 (float32: 2^-24) with float32's exponent range.  Read when a
    function is streamed.  On by default.
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
    """FINUFFT plans made and not yet destroyed; zero once every owner of one has been freed."""
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
    """Normal operators built since the last reset, by point spread function and by transform pair.

    ``pics --no-toeplitz`` and ``nufft -t`` decide which.
    """
    from bartorch._lib import library

    lib = library()
    return int(lib.bartorch_toeplitz_counter(0)), int(lib.bartorch_toeplitz_counter(1))


def functions_compressed() -> int:
    """Toeplitz functions built compressed since the last reset; decided at build time."""
    from bartorch._lib import library

    return int(library().bartorch_toeplitz_counter(2))


def functions_real() -> int:
    """Toeplitz functions stored as floats since the last reset; decided from the basis."""
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
    """Reset the NUFFT and Toeplitz counters."""
    from bartorch._lib import library

    lib = library()
    lib.bartorch_nufft_reset_counters()
    lib.bartorch_toeplitz_reset_counters()


def tolerance() -> float:
    """The tolerance FINUFFT plans are made with."""
    from bartorch._lib import library

    return float(library().bartorch_finufft_tolerance())


def upsampling() -> float:
    """Oversampling of FINUFFT's fine grid; 0 lets FINUFFT choose.

    BART's ``-o`` overrides it wherever it is not BART's default of 2.
    """
    from bartorch._lib import library

    return float(library().bartorch_finufft_upsampling())


def set_threads(n: int) -> None:
    """Set the threads a host transform takes; 0 lets FINUFFT take one per physical core.

    :func:`bartorch.set_num_threads` sets this together with BART's own count.
    Ignored on a card.
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
    """Whether ``bart nufft`` answers the same with FINUFFT and with BART's gridder.

    Relative to BART's peak; ``tolerance`` allows for BART's gridding error.
    """
    import bartorch
    import bartorch.tools as bt
    from bartorch._lib import library

    lib = library()
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    image = bt.phantom([n, n]).reshape(1, n, n)

    lib.bartorch_finufft_use_in_tools(1)
    fast = bartorch.nufft(image, traj)

    with barts_own_gridder():
        reference = bartorch.nufft(image, traj)

    lib.bartorch_finufft_use_in_tools(1)

    scale = reference.abs().max()
    if scale == 0:
        return False
    return bool(((fast - reference).abs().max() / scale).item() < tolerance)


def three_components(traj: torch.Tensor) -> torch.Tensor:
    """``traj`` with ``kz`` written out as zero where it carries ``kx, ky`` only.

    BART's operators take three components per sample, and the FINUFFT
    substitution reads them in that layout.  The FINUFFT plan's dimension
    comes from the image, so a two-dimensional image is transformed by a 2D
    plan either way.  The padding is one copy of the trajectory, made when an
    operator is built; a three-component trajectory is returned as it is.
    """
    d = int(traj.shape[-1])
    if d == 3:
        return traj
    if d != 2:
        raise ValueError(f"a trajectory carries 2 or 3 components per sample, not {d}")
    return torch.cat((traj, torch.zeros_like(traj[..., :1])), dim=-1).contiguous()


def spatial_ndim(traj: torch.Tensor) -> int:
    """2 or 3: whether the trajectory's third component is used (BART always carries three)."""
    if traj.shape[-1] < 3:
        return 2
    return 3 if bool(torch.any(traj[..., 2].real != 0)) else 2


@contextlib.contextmanager
def barts_own_gridder():
    """Context in which BART's Kaiser-Bessel gridder serves the NUFFT instead of FINUFFT.

    For tests and the install-time agreement check only; nothing public reaches it.
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
    """Install the substitution on first use; a failure is logged once at warning level, not raised.

    Most commands need no NUFFT.  A NUFFT asked for after a failed install is
    refused, with the reason.
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
        logging.getLogger("bartorch._finufft").warning(
            "FINUFFT is installed but was not put in BART's place, so its transforms "
            "will be BART's own gridder: %s",
            exc,
        )
