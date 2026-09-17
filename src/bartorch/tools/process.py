"""Operations on an image rather than on k-space: resampling, registration, measurement.

Each is one of BART's array applications.  None is a reconstruction and none is
part of an encoding: a reconstruction is surrounded by them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from bartorch import _grid
from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = [
    "affine_transform",
    "estimate_shift",
    "fovshift",
    "mse",
    "nrmse",
    "psnr",
    "register_affine",
    "register_nonrigid",
    "roi_stat",
    "ssim",
    "warp",
]

_TRANSFORMS = {"translation": "T", "rigid": "R", "affine": "A"}

_STATS = {"count": "C", "sum": "S", "mean": "M", "std": "D", "energy": "E", "variance": "V"}


@curated("interpolate")
def warp(
    input: torch.Tensor,
    displacement: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    order: int = 1,
) -> torch.Tensor:
    """Pull ``input`` through a displacement field: ``out[p] = input(p + displacement[p])``.

    ``p`` and the displacement are in voxels along ``axes``; the grid is
    zero outside ``[0, n - 1]``.  BART's ``interpolate -D``.

    Parameters
    ----------
    input : torch.Tensor
    displacement : torch.Tensor
        Real field of shape ``(..., len(axes))``; component ``i`` of the last
        axis is the displacement along ``axes[i]``.  The leading shape is
        aligned right against ``input.shape``, equal to it along ``axes`` and
        one or equal elsewhere.
    axes : int or tuple of int
        The last ``len(axes)`` axes, in any order: BART interpolates along the
        first ``len(axes)`` BART dims whatever the flags
        (motion/displacement.c:117).
    order : {0, 1, 3}
        Nearest neighbour, linear, or Keys cubic; see :func:`interpolate`.

    Returns
    -------
    torch.Tensor
        Same shape as ``input``.
    """
    ndim = input.ndim
    axes = _grid.normalize_axes(axes, ndim)
    if set(axes) != set(range(ndim - len(axes), ndim)):
        raise ValueError(f"warp axes must be the last {len(axes)} axes, got {axes}")
    _grid.check_motion_dim(input, axes)
    field, grid = _grid.motion_field(displacement, ndim, axes, "displacement")
    for ax in range(ndim):
        size, have = grid[ax], input.shape[ax]
        if size != have and (ax in axes or size != 1):
            raise ValueError(
                f"displacement has size {size} along axis {ax}, the input {have}; "
                + ("they must match" if ax in axes else "it must be one or match")
            )
    out = dispatch(
        "interpolate",
        [input, field],
        None,
        _pos=[axes_flags(axes, ndim)],
        D=True,
        **_grid.order_flags(order),
    )
    return out.reshape(input.shape)


@curated("interpolate")
def affine_transform(
    input: torch.Tensor,
    matrix: torch.Tensor,
    axes: tuple[int, ...],
    oshape: tuple[int, ...] | None = None,
    *,
    order: int = 1,
) -> torch.Tensor:
    """Resample ``input`` on an affinely mapped grid: ``out(u) = input(matrix @ [u, 1])``.

    The matrix maps output positions to input positions.  Positions are in
    units of each grid's own field of view with the origin at index
    ``n // 2``: voxel ``p`` of a grid of ``n`` sits at ``(p - n // 2) / n``
    (motion/affine.c:25, 152, 466).  A translation of ``t`` therefore moves
    the content by ``-t * n`` voxels, and the identity with an ``oshape``
    other than the input's rescales the whole field of view onto the new
    grid.  The grid is zero outside ``[0, n - 1]``.  BART's
    ``interpolate -A``, which is limited to three dims.

    Parameters
    ----------
    input : torch.Tensor
        Two or three spatial axes, optionally after leading axes of size one.
    matrix : torch.Tensor
        Real ``(k, k + 1)`` or homogeneous ``(k + 1, k + 1)`` matrix,
        ``k = len(axes)``, acting on column vectors ``[u_0, ..., u_{k-1}, 1]``
        whose entry ``i`` is the position along ``axes[i]``.
    axes : tuple of int
        The last two or three axes, in the order the matrix's rows and
        columns refer to them.
    oshape : tuple of int, optional
        Output size of the last ``k`` axes, in C order; by default
        the input's.

    Returns
    -------
    torch.Tensor
        ``input.shape`` with its last ``k`` axes of size ``oshape``.

    Examples
    --------
    Swap the two axes of a square image (its transpose):

    >>> affine_transform(image, torch.tensor([[0.0, 1, 0], [1, 0, 0]]), axes=(-2, -1))
    """
    ndim = input.ndim
    axes = _grid.normalize_axes(axes, ndim)
    k = len(axes)
    if k not in (2, 3) or set(axes) != set(range(ndim - k, ndim)):
        raise ValueError(f"affine axes must be the last two or three axes, got {axes}")
    if any(n != 1 for n in input.shape[: ndim - k]):
        raise ValueError(
            "BART's affine interpolation covers three dims only; "
            f"the leading axes of {tuple(input.shape)} must be of size one"
        )
    m = torch.as_tensor(matrix)
    m = (m.real if m.is_complex() else m).to(torch.float64).cpu().numpy()
    if m.shape == (k + 1, k + 1):
        if not np.allclose(m[k], np.eye(k + 1)[k]):
            raise ValueError(f"the last row of a homogeneous matrix must be {np.eye(k + 1)[k]}")
        m = m[:k]
    if m.shape != (k, k + 1):
        raise ValueError(f"matrix must be ({k}, {k + 1}) or ({k + 1}, {k + 1}), got {m.shape}")
    # BART's 3x4 matrix, indexed by BART dims (x, y, z) and stored Fortran
    # order, which in C order is its transpose.
    a = np.eye(3, 4)
    for r, ar in enumerate(axes):
        br = ndim - 1 - ar
        for c, ac in enumerate(axes):
            a[br, ndim - 1 - ac] = m[r, c]
        a[br, 3] = m[r, k]
    bart_matrix = torch.from_numpy(np.ascontiguousarray(a.T)).to(torch.complex64)
    spatial = tuple(input.shape[ndim - k :]) if oshape is None else tuple(int(n) for n in oshape)
    if len(spatial) != k:
        raise ValueError(f"oshape must have {k} entries, got {spatial}")
    x = tuple(reversed(spatial)) + (1,) * (3 - k)
    out = dispatch(
        "interpolate",
        [input, bart_matrix],
        None,
        _pos=[7],
        A=True,
        x=x,
        **_grid.order_flags(order),
    )
    return out.reshape(*input.shape[: ndim - k], *spatial)


@curated("fovshift")
def fovshift(
    input: torch.Tensor,
    shift: Sequence[float],
    *,
    traj: torch.Tensor | None = None,
    pixels: bool = False,
) -> torch.Tensor:
    """Shift the field of view of k-space data by a linear phase.

    Multiplies ``input`` by ``exp(2 pi i k . shift)``, with ``k`` in cycles
    per field of view and ``shift`` in fields of view (fovshift.c:31, 128;
    num/filter.c:256).  On Cartesian data ``k = p - n / 2`` for index ``p``
    of the last three axes -- zero at the centre of a centred FFT for even
    ``n``.  The image moves by ``-shift``: under :func:`bartorch.fft`'s
    convention the result is the spectrum of ``image[p + shift * n]``.

    Parameters
    ----------
    input : torch.Tensor
        Centred Cartesian k-space whose last three axes (or fewer) are
        ``(kz, ky, kx)``, or with ``traj`` the samples shaped as
        :func:`bartorch.nufft` returns them.
    shift : sequence of float
        Shift along the image axes in C order, ``(z, y, x)`` or ``(y, x)``
        or ``(x,)``; in fields of view, or in voxels with ``pixels``.
    traj : torch.Tensor, optional
        Trajectory ``(..., samples, 3)`` in grid units, ``kx, ky, kz``, as
        :func:`bartorch.tools.traj` produces.
    pixels : bool
        ``shift`` in voxels rather than fields of view; Cartesian
        only.
    """
    shift = [float(s) for s in shift]
    if not 1 <= len(shift) <= 3:
        raise ValueError(f"shift has one to three entries, got {len(shift)}")
    if pixels and traj is not None:
        raise ValueError("a shift in pixels is only defined for Cartesian k-space")
    if traj is None and len(shift) > input.ndim:
        raise ValueError(f"shift has {len(shift)} entries for an input of {input.ndim} axes")
    s = tuple(reversed(shift)) + (0.0,) * (3 - len(shift))
    out = dispatch("fovshift", [input], None, s=s, t=traj, p=pixels)
    return out.reshape(input.shape)


@curated("affinereg")
def register_affine(
    reference: torch.Tensor,
    moved: torch.Tensor,
    *,
    transform: str = "rigid",
    reference_mask: torch.Tensor | None = None,
    moved_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Affine transform that maps ``reference`` onto ``moved``, by mutual information.

    Registers magnitudes over a three-level Gaussian pyramid.

    Parameters
    ----------
    reference, moved : torch.Tensor
        Images of C shape ``(z, y, x)`` or ``(y, x)``; any further leading
        axis must have size one.  A ``reference`` without ``z``, or with
        ``z`` of size one, is registered in two dimensions.
    transform : {"translation", "rigid", "affine"}
        Degrees of freedom: translation only, rotation and
        translation, or all.
    reference_mask, moved_mask : torch.Tensor, optional
        Binary masks shaped like their image; both or neither.

    Returns
    -------
    torch.Tensor
        Real ``(k, k + 1)`` matrix ``[A | t]``, ``k`` two or three, whose rows
        and columns are the last ``k`` axes in C order.  It maps a position in
        ``reference`` to the position in ``moved`` that lands there,
        ``p_moved = A @ p_ref + t``, each in its own grid's field-of-view
        units ``(index - n // 2) / n`` (``motion/affine.c:152-171``,
        ``:428-465``): content of ``moved`` displaced by ``d`` voxels along
        the last axis gives ``t[-1] = d / n``.  This is the convention
        :func:`bartorch.tools.affine_transform` takes:
        ``affine_transform(moved, matrix, axes=tuple(range(-k, 0)))``
        resamples ``moved`` onto ``reference``.
    """
    try:
        flag = _TRANSFORMS[transform]
    except KeyError:
        raise ValueError(
            f"transform must be one of {sorted(_TRANSFORMS)}, got {transform!r}"
        ) from None
    for name, image in (("reference", reference), ("moved", moved)):
        if any(n != 1 for n in image.shape[:-3]):
            raise ValueError(
                f"{name} has shape {tuple(image.shape)}; only the last three axes may exceed one"
            )
    if (reference_mask is None) != (moved_mask is None):
        raise ValueError("give both masks or neither")
    if reference_mask is not None:
        _grid.same_shape(reference, reference_mask, "reference and reference_mask")
        _grid.same_shape(moved, moved_mask, "moved and moved_mask")
    stored = dispatch(
        "affinereg",
        [reference, moved],
        None,
        mask_reference=reference_mask,
        mask_moved=moved_mask,
        **{flag: True},
    )
    # BART's 3x4 matrix over its dims (x, y, z), stored Fortran order: in C
    # order that is the (4, 3) transpose.  BART registers in 2D when the
    # reference's z is one (affinereg.c:114-122), leaving that row the identity.
    bart = stored.real.T
    k = 2 if reference.ndim < 3 or reference.shape[-3] == 1 else 3
    rows = [k - 1 - r for r in range(k)]
    matrix = torch.empty(k, k + 1, dtype=bart.dtype)
    matrix[:, :k] = bart[rows][:, rows]
    matrix[:, k] = bart[rows, 3]
    return matrix


@curated("estmotion")
def register_nonrigid(
    reference: torch.Tensor,
    moved: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    levels: int | None = None,
    optical_flow: bool = False,
    tv_weight: float | None = None,
    max_flow: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Displacement field from ``reference`` into ``moved``, by greedy SyN or TV-L1 optical flow.

    Registers magnitudes.

    Parameters
    ----------
    reference, moved : torch.Tensor
        Images of the same shape, at most eight axes.
    axes : int or tuple of int
        Axes to register along.
    levels : int, optional
        Gaussian pyramid levels, one to five; BART's default is three.
    optical_flow : bool
        TV-L1 optical flow instead of greedy SyN.
    tv_weight : float, optional
        TV regularization of the optical flow; BART's default is 0.01.
    max_flow : float, optional
        Bound on the flow magnitude of the optical flow.

    Returns
    -------
    field : torch.Tensor
        Real, shape ``(*reference.shape, len(axes))``; component ``i`` is the
        displacement in voxels along ``axes[i]``, such that
        ``moved(p + field[p]) ≈ reference(p)``.  What :func:`bartorch.tools.warp`
        takes: ``warp(moved, field, axes)`` resamples ``moved`` onto
        ``reference``.
    inverse_field : torch.Tensor or None
        Same layout, ``reference(p + inverse_field[p]) ≈ moved(p)``.  ``None``
        with ``optical_flow``: BART computes no inverse there
        (``estmotion.c:275``).
    """
    _grid.same_shape(reference, moved, "reference and moved")
    ndim = reference.ndim
    if ndim > _grid.MOTION_DIM:
        raise ValueError(f"at most {_grid.MOTION_DIM} axes, got {ndim}")
    axes = _grid.normalize_axes(axes, ndim)
    if levels is not None and not 1 <= levels <= 5:
        raise ValueError(f"levels must be between 1 and 5, got {levels}")
    if not optical_flow and (tv_weight is not None or max_flow is not None):
        raise ValueError("tv_weight and max_flow apply only with optical_flow=True")
    out = dispatch(
        "estmotion",
        [reference, moved],
        None,
        _pos=[axes_flags(axes, ndim)],
        _n_out=1 if optical_flow else 2,
        l=levels,
        optical_flow=optical_flow,
        r=tv_weight,
        max_flow=max_flow,
    )

    # BART keeps the components on its MOTION_DIM, ahead of the image in C order.
    def as_field(f: torch.Tensor) -> torch.Tensor:
        f = f.reshape(len(axes), *reference.shape).real
        return _grid.from_bart_order(f, axes).movedim(0, -1).contiguous()

    if optical_flow:
        return as_field(out), None
    return as_field(out[0]), as_field(out[1])


@curated("estshift")
def estimate_shift(
    a: torch.Tensor,
    b: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    fov_units: bool = False,
) -> torch.Tensor:
    """Sub-voxel shift that moves ``b`` onto ``a``, from the phase of their cross-spectrum.

    Parameters
    ----------
    a, b : torch.Tensor
        Arrays of the same shape.
    axes : int or tuple of int
        Axes to estimate the shift along.
    fov_units : bool
        Shift as a fraction of the axis length rather than in voxels.

    Returns
    -------
    torch.Tensor
        Real, one shift per entry of ``axes`` in that order:
        ``a ≈ torch.roll(b, shifts, axes)`` for whole-voxel shifts.
    """
    _grid.same_shape(a, b, "a and b")
    axes = _grid.normalize_axes(axes, a.ndim)
    shifts = dispatch("estshift", [a, b], None, _pos=[axes_flags(axes, a.ndim)], f=fov_units)
    return _grid.from_bart_order(shifts.reshape(-1).real, axes).contiguous()


@curated("nrmse")
def nrmse(reference: torch.Tensor, input: torch.Tensor, *, scaled: bool = False) -> float:
    """Normalized root mean square error ``‖input - reference‖ / ‖reference‖``.

    Parameters
    ----------
    scaled : bool
        First scale ``reference`` by the complex least-squares factor
        ``Σ conj(reference)·input / ‖reference‖²``.
    """
    _grid.same_shape(reference, input, "reference and input")
    return float(dispatch("nrmse", [reference, input], False, s=scaled, scientific=True))


def _measure(option: str, reference: torch.Tensor, input: torch.Tensor) -> float:
    _grid.same_shape(reference, input, "reference and input")
    return float(dispatch("measure", [reference, input], False, **{option: True}))


@curated("measure")
def mse(reference: torch.Tensor, input: torch.Tensor, *, magnitude: bool = False) -> float:
    """Mean squared error ``mean |input - reference|²`` over every element .

    Parameters
    ----------
    magnitude : bool
        Compare magnitudes, ``mean (|input| - |reference|)²``, taken as the
        root sum of squares over the coil axis ``-4`` of a C-order
        ``(..., coils, z, y, x)`` array .
    """
    return _measure("mse_mag" if magnitude else "mse", reference, input)


@curated("measure")
def ssim(reference: torch.Tensor, input: torch.Tensor) -> float:
    """Mean structural similarity of the magnitudes .

    Magnitudes are the root sum of squares over the coil axis, and each image
    is divided by the maximum magnitude of its ``reference``.  SSIM with
    ``k1 = 0.01``, ``k2 = 0.03``, dynamic range one and population
    statistics over a uniform 7x7 window in ``(y, x)``, at every position the
    window fits whole (no padding), averaged over those positions, over ``z``
    and over every other axis (``networks/losses.c:304-327``,
    ``nn/losses.c:151-285``).

    Parameters
    ----------
    reference, input : torch.Tensor
        C order ``(..., coils, z, y, x)``; a plain ``(y, x)`` image serves.
        ``y`` and ``x`` are at least seven.
    """
    if min(reference.shape[-2:], default=0) < 7:
        raise ValueError(f"ssim needs y and x of at least 7, got shape {tuple(reference.shape)}")
    return _measure("ssim", reference, input)


@curated("measure")
def psnr(reference: torch.Tensor, input: torch.Tensor) -> float:
    """Peak signal-to-noise ratio of the magnitudes in decibels, averaged over images .

    Per image over ``(z, y, x)``: ``20 log10 max|reference| - 10 log10
    mean (|input| - |reference|)²``, magnitudes being the root sum of squares
    over the coil axis; then the mean over every other axis
    (``nn/losses.c:54-100``).

    Parameters
    ----------
    reference, input : torch.Tensor
        C order ``(..., coils, z, y, x)``; a plain ``(y, x)`` image serves.
    """
    return _measure("psnr", reference, input)


@curated("roistat")
def roi_stat(
    roi: torch.Tensor, input: torch.Tensor, stat: str = "mean", *, bessel: bool = False
) -> torch.Tensor:
    """Statistic of ``input`` over the region ``roi``.

    Parameters
    ----------
    roi : torch.Tensor
        Weights, binary for the plain statistics.  Broadcasts against
        ``input``; an axis where ``roi`` has size one and ``input`` does not
        (or the reverse, several regions along it) is kept.
    stat : {"count", "sum", "mean", "std", "energy", "variance"}
        ``count`` is the sum of the weights; ``sum`` of the weighted
        values; ``mean`` their ratio; ``energy`` the sum of
        squared deviations from the mean, not of squared values;
        ``variance`` energy over count; ``std`` its root.
    bessel : bool
        Divide by count minus one; ``std`` and ``variance`` only.

    Returns
    -------
    torch.Tensor
        Broadcast shape of ``roi`` and ``input``, with size one on the axes
        where both exceed one.  Complex for ``sum`` and ``mean``, real
        otherwise.
    """
    try:
        flag = _STATS[stat]
    except KeyError:
        raise ValueError(f"stat must be one of {sorted(_STATS)}, got {stat!r}") from None
    if bessel and stat not in ("std", "variance"):
        raise ValueError("bessel applies only to std and variance")
    try:
        shape = torch.broadcast_shapes(tuple(roi.shape), tuple(input.shape))
    except RuntimeError:
        raise ValueError(
            f"roi of shape {tuple(roi.shape)} does not broadcast "
            f"against input of shape {tuple(input.shape)}"
        ) from None
    ndim = len(shape)
    rshape = (1,) * (ndim - roi.ndim) + tuple(roi.shape)
    ishape = (1,) * (ndim - input.ndim) + tuple(input.shape)
    oshape = tuple(1 if (r > 1 and i > 1) else n for r, i, n in zip(rshape, ishape, shape))
    out = dispatch("roistat", [roi, input], None, b=bessel, **{flag: True}).reshape(oshape)
    return out if stat in ("sum", "mean") else out.real.contiguous()
