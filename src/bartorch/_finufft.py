"""FINUFFT as a BART operator.

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


def install_gridder(enable: bool = True, tolerance: float = 1e-6) -> bool:
    """Put FINUFFT underneath BART's own gridder, for every tool that grids.

    BART's ``pics``, ``nlinv``, ``moba`` and ``nufft`` all grid through the
    same two routines and deapodise with the transform of the kernel those
    routines used.  Both are replaced together here, so the pair stays a
    transform; anything the replacement cannot serve falls back to BART's
    Kaiser-Bessel gridder.

    Returns whether FINUFFT is in place.
    """
    from bartorch._lib import library

    lib = library()
    if not enable:
        lib.bartorch_finufft_enable(0)
        return False

    path = _library_path()
    if path is None:
        return False

    import ctypes as c

    from finufft._finufft import FinufftOpts

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

    # The options struct is version-dependent, so its layout is read from the
    # package that will interpret it rather than assumed here.
    if (
        lib.bartorch_finufft_layout(
            c.sizeof(FinufftOpts),
            FinufftOpts.modeord.offset,
            FinufftOpts.spreadinterponly.offset,
            FinufftOpts.upsampfac.offset,
            FinufftOpts.nthreads.offset,
        )
        != 0
    ):
        return False

    lib.bartorch_finufft_set_tolerance(float(tolerance))
    lib.bartorch_finufft_enable(1)

    # BART grids, transforms and deapodises as one; a kernel swapped into the
    # middle of that is only right if the deapodisation is the transform of
    # the kernel that was actually used, and getting that wrong is quiet
    # rather than loud.  So the substitution has to prove itself against
    # BART's own gridder before it is left in place.
    if not _agrees_with_bart():
        lib.bartorch_finufft_enable(0)
        return False

    return bool(lib.bartorch_finufft_active())


def _agrees_with_bart(tolerance: float = 5e-3) -> bool:
    """Whether the substituted gridder computes what BART's gridder computes."""
    import bartorch.tools as bt
    from bartorch._lib import library

    lib = library()
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    image = bt.phantom([n, n]).reshape(1, n, n)

    lib.bartorch_finufft_enable(1)
    fast = bt.nufft(traj, image)
    lib.bartorch_finufft_enable(0)
    reference = bt.nufft(traj, image)
    lib.bartorch_finufft_enable(1)

    scale = reference.abs().max()
    if scale == 0:
        return False
    return bool(((fast - reference).abs().max() / scale).item() < tolerance)


def gridder_active() -> bool:
    """Whether BART's tools are gridding with FINUFFT."""
    from bartorch._lib import library

    return bool(library().bartorch_finufft_active())


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
