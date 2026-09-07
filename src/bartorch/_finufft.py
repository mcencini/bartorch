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


def _library_path() -> str | None:
    """The compiled library inside the ``finufft`` wheel."""
    try:
        import finufft
    except ImportError:
        return None
    for name in ("libfinufft.so", "libfinufft.dylib", "finufft.dll", "libfinufft.dll"):
        candidate = Path(finufft.__file__).resolve().parent / name
        if candidate.exists():
            return str(candidate)
    return None


def _load_symbols() -> bool:
    """Hand the library FINUFFT's entry points and its options layout.

    The layout is read from the package that will interpret the struct, so a
    release that moves a field cannot be misread here.
    """
    import ctypes as c

    from bartorch._lib import library

    lib = library()
    path = _library_path()
    if path is None:
        return False

    try:
        from finufft._finufft import FinufftOpts
    except ImportError:
        return False

    handle = c.CDLL(path)
    _keepalive.append(handle)
    for symbol in (
        "finufftf_makeplan",
        "finufftf_setpts",
        "finufftf_execute",
        "finufftf_destroy",
        "finufftf_default_opts",
    ):
        try:
            fn = getattr(handle, symbol)
        except AttributeError:
            return False
        if lib.bartorch_finufft_set(symbol.encode(), c.cast(fn, c.c_void_p)) != 0:
            return False

    return 0 == lib.bartorch_finufft_layout(c.sizeof(FinufftOpts), FinufftOpts.nthreads.offset)


def use_in_tools(enable: bool = True, tolerance: float = 1e-6) -> bool:
    """Have BART's own tools compute their NUFFT with FINUFFT.

    Returns whether the substitution is in place and agrees with BART.
    """
    from bartorch._lib import library

    lib = library()
    if not enable:
        lib.bartorch_finufft_use_in_tools(0)
        return False

    if not _load_symbols():
        return False

    lib.bartorch_finufft_set_tolerance(float(tolerance))
    lib.bartorch_finufft_use_in_tools(1)

    if not _tools_agree_with_bart():
        lib.bartorch_finufft_use_in_tools(0)
        return False

    return bool(lib.bartorch_finufft_usable())


def used_in_tools() -> bool:
    """Whether BART's tools are computing their NUFFT with FINUFFT."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_usable())


_DECLINED = {
    0: "",
    1: "FINUFFT is not in use",
    2: "the trajectory is missing",
    3: "the trajectory is on a device",
    4: "the trajectory does not carry three components",
    5: "k-space is not a single line of samples per readout",
    6: "the transform is over axes other than the spatial three",
    7: "the image has no spatial extent",
    8: "the trajectory and k-space disagree on the number of samples",
    9: "k-space and the coil images disagree beyond the spatial axes",
    10: "there are too many frames for one plan",
    11: "FINUFFT would not plan the forward transform",
    12: "FINUFFT would not plan the adjoint transform",
    13: "FINUFFT would not take the trajectory",
    14: "a subspace basis is in use",
    15: "the weights do not lie along k-space",
}


def decline_reason() -> str:
    """Why the last operator was BART's rather than FINUFFT's; empty if it was FINUFFT's."""
    from bartorch._lib import library

    code = library().bartorch_nufft_decline_reason()
    return _DECLINED.get(code, f"reason {code}")


def operators_built() -> tuple[int, int]:
    """NUFFT operators built since the last reset, by FINUFFT and by BART."""
    from bartorch._lib import library

    lib = library()
    return int(lib.bartorch_nufft_counter(0)), int(lib.bartorch_nufft_counter(1))


def reset_counters() -> None:
    """Start counting operators again."""
    from bartorch._lib import library

    library().bartorch_nufft_reset_counters()


def tolerance() -> float:
    """The tolerance FINUFFT plans are made with."""
    from bartorch._lib import library

    return float(library().bartorch_finufft_tolerance())


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
    lib.bartorch_finufft_use_in_tools(0)
    reference = bt.nufft(traj, image)
    lib.bartorch_finufft_use_in_tools(1)

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
