"""FINUFFT, as an operator and underneath BART's own tools.

The ``finufft`` and ``cufinufft`` wheels carry a compiled library and a thin
Python wrapper over it, so a plan made here is the same object BART would
have made for itself, and the transform runs with the GIL released.  A plan is
made once per operator and reused, which is what makes this worth doing inside
an iterative solve.

The scaling and sign follow BART's own NUFFT, so an operator from here is
interchangeable with :meth:`bartorch.ops.LinearOperator.nufft`: a type-2
transform with a negative exponent, divided by the square root of the number
of voxels.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
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
        opts, field, upsampling = _options_layout(package)
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

    return 0 == lib.bartorch_finufft_layout(device, c.sizeof(opts), field.offset, upsampling.offset)


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
    to spread, which they spell alike.
    """
    if package == "finufft":
        from finufft._finufft import FinufftOpts as opts

        return opts, opts.nthreads, opts.upsampfac
    from cufinufft._cufinufft import NufftOpts as opts

    return opts, opts.gpu_device_id, opts.upsampfac


def use_in_tools(
    enable: bool = True,
    tolerance: float = 1e-6,
    fallback: bool = False,
    upsampling: float = 0.0,
) -> bool:
    """Have BART's own tools compute their NUFFT with FINUFFT.

    Parameters
    ----------
    enable : bool
        Turn the substitution on, or off to leave BART its own gridder.
    tolerance : float
        The tolerance FINUFFT plans are made with.
    upsampling : float
        How far past the image to spread before transforming.  Two is the
        textbook grid; a quarter over trades a smaller one for a wider kernel,
        which pays only where the samples are sparse enough that spreading is
        not what the transform spends its time in.  Zero, the default, lets
        FINUFFT weigh that per problem.  BART's ``-o`` takes precedence
        wherever it is not BART's own default of two.
    fallback : bool
        Whether BART's own operator may answer a transform FINUFFT cannot
        serve.  By default it may not: such a transform raises, naming the
        reason, rather than reconstructing more slowly and less accurately
        without saying so.

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
        raise ImportError("this needs the finufft package: pip install 'bartorch[finufft]'")

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

    lib.bartorch_finufft_set_tolerance(float(tolerance))
    lib.bartorch_finufft_set_upsampling(float(upsampling))
    lib.bartorch_nufft_allow_fallback(int(bool(fallback)))
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

    BART's ``-o`` is the same number and takes precedence wherever it is not
    BART's own default of two; zero leaves the choice to FINUFFT.
    """
    from bartorch._lib import library

    return float(library().bartorch_finufft_upsampling())


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

    # BART's own operator, asked for rather than fallen back on.
    allowed = lib.bartorch_nufft_fallback_allowed()
    lib.bartorch_nufft_allow_fallback(1)
    lib.bartorch_finufft_use_in_tools(0)
    reference = bt.nufft(traj, image)
    lib.bartorch_finufft_use_in_tools(1)
    lib.bartorch_nufft_allow_fallback(allowed)

    scale = reference.abs().max()
    if scale == 0:
        return False
    return bool(((fast - reference).abs().max() / scale).item() < tolerance)


def _coordinates(traj: torch.Tensor, spatial: Shape) -> list[np.ndarray]:
    """BART trajectory in grid samples to FINUFFT's radians, slowest axis first.

    A BART trajectory holds kx, ky, kz along its last axis, kx belonging to the
    fastest-varying image axis; FINUFFT takes one coordinate array per image
    axis in the order the image is laid out, so the order is reversed here.
    """
    coords = traj.detach().cpu().numpy().real.astype(np.float32)
    if coords.shape[-1] < len(spatial):
        raise ValueError(f"trajectory has {coords.shape[-1]} components, need {len(spatial)}")
    out = []
    for axis, size in enumerate(spatial):
        component = coords[..., len(spatial) - 1 - axis].ravel()
        out.append(np.ascontiguousarray(2 * np.pi * component / size, dtype=np.float32))
    return out


class _Transforms:
    """A pair of FINUFFT plans, made once and reused by both callbacks."""

    def __init__(self, traj: torch.Tensor, spatial: Shape, batch: int, eps: float, scale: float):
        import finufft

        self.spatial = tuple(spatial)
        self.batch = int(batch)
        self.scale = float(scale)
        self.samples = int(traj.numel() // traj.shape[-1])
        coords = _coordinates(traj, self.spatial)
        common = dict(n_trans=self.batch, eps=eps, dtype="complex64")
        self.forward_plan = finufft.Plan(2, self.spatial, isign=-1, **common)
        self.adjoint_plan = finufft.Plan(1, self.spatial, isign=+1, **common)
        self.forward_plan.setpts(*coords)
        self.adjoint_plan.setpts(*coords)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = np.ascontiguousarray(image.detach().cpu().numpy().reshape(self.batch, *self.spatial))
        y = self.forward_plan.execute(x)
        return torch.from_numpy(np.ascontiguousarray(y * self.scale))

    def adjoint(self, samples: torch.Tensor) -> torch.Tensor:
        y = np.ascontiguousarray(
            samples.detach().cpu().numpy().reshape(self.batch, self.samples).astype(np.complex64)
        )
        x = self.adjoint_plan.execute(y)
        return torch.from_numpy(np.ascontiguousarray(x * self.scale))


def spatial_ndim(traj: torch.Tensor) -> int:
    """Whether a trajectory is two- or three-dimensional.

    A BART trajectory always carries three components; a two-dimensional one
    leaves the third at zero.
    """
    if traj.shape[-1] < 3:
        return 2
    return 3 if bool(torch.any(traj[..., 2].real != 0)) else 2


def transforms(
    traj: torch.Tensor,
    image_shape: Shape,
    eps: float = 1e-6,
    scale: float | None = None,
    ndim: int | None = None,
) -> tuple[_Transforms, Shape]:
    """Plans for *image_shape*, and the sample shape they produce.

    Parameters
    ----------
    traj : tensor
        Trajectory in grid samples, ``(..., samples, 3)``, as
        ``bartorch.tools.traj`` produces.
    image_shape : tuple of int
        Coil-image shape in C order, the last two or three axes spatial.
    eps : float
        FINUFFT's tolerance.
    scale : float, optional
        By default the factor that makes this agree with BART's own NUFFT.
    ndim : int, optional
        Spatial dimensions; by default two or three as the trajectory says.
    """
    if not available():
        raise ImportError("this needs the finufft package: pip install 'bartorch[finufft]'")
    image_shape = tuple(image_shape)
    if ndim is None:
        ndim = spatial_ndim(traj)
    if len(image_shape) < ndim:
        raise ValueError(f"a {ndim}-dimensional transform needs at least that many image axes")
    spatial = image_shape[-ndim:]
    batch = math.prod(image_shape[:-ndim]) if len(image_shape) > ndim else 1
    if scale is None:
        scale = 1.0 / math.sqrt(math.prod(spatial))
    plans = _Transforms(traj, spatial, batch, eps, scale)
    kspace_shape = tuple(image_shape[:-ndim]) + tuple(traj.shape[:-1]) + (1,)
    return plans, kspace_shape


def disable() -> None:
    """Compute every NUFFT with BART's own gridder instead.

    Nothing reaches it by accident, so taking it for everything is something
    to say out loud.
    """
    use_in_tools(False)


_installed = False


def install_once() -> None:
    """Put the substitution in place the first time anything needs it.

    A caller who has the package should not have to ask for it, and one who
    does not should hear about it when a transform wants it rather than get a
    quieter answer from BART.  What went wrong is left to the transform to
    report, because most of what BART does needs no NUFFT at all.
    """
    global _installed
    if _installed:
        return
    _installed = True
    if not available():
        return
    try:
        use_in_tools(True)
    except (ImportError, RuntimeError):
        pass
