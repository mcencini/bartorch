"""Image registration."""

from __future__ import annotations

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["estimate_shift", "register_affine", "register_nonrigid"]

_TRANSFORMS = {"translation": "T", "rigid": "R", "affine": "A"}

# BART's MOTION_DIM (ITER_DIM, misc/mri.h): a displacement field's components
# lie along it, so an image must be singleton there and C order can carry at
# most eight axes.
_MOTION_DIM = 8


def _check_same_shape(a: torch.Tensor, b: torch.Tensor, names: str) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"{names} must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}"
        )


def _normalize_axes(axes: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    axes = (axes,) if isinstance(axes, int) else tuple(int(a) for a in axes)
    if any(not -ndim <= a < ndim for a in axes):
        raise ValueError(f"axes {axes} out of range for {ndim} axes")
    out = tuple(a % ndim for a in axes)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {axes}")
    return out


def _from_bart_order(components: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    """Reorder a leading component axis from BART's (descending C axis) to that of ``axes``."""
    descending = sorted(axes, reverse=True)
    return components[[descending.index(a) for a in axes]]


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
        Degrees of freedom: translation only (``-T``), rotation and
        translation (``-R``), or all (``-A``).
    reference_mask, moved_mask : torch.Tensor, optional
        Binary masks shaped like their image (``--mask-reference``,
        ``--mask-moved``); both or neither.

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
        :func:`bartorch.affine_transform` takes:
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
        _check_same_shape(reference, reference_mask, "reference and reference_mask")
        _check_same_shape(moved, moved_mask, "moved and moved_mask")
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
        Gaussian pyramid levels, one to five (``-l``); BART's default is three.
    optical_flow : bool
        TV-L1 optical flow instead of greedy SyN (``--optical-flow``).
    tv_weight : float, optional
        TV regularization of the optical flow (``-r``); BART's default is 0.01.
    max_flow : float, optional
        Bound on the flow magnitude of the optical flow (``--max-flow``).

    Returns
    -------
    field : torch.Tensor
        Real, shape ``(*reference.shape, len(axes))``; component ``i`` is the
        displacement in voxels along ``axes[i]``, such that
        ``moved(p + field[p]) ≈ reference(p)``.  What :func:`bartorch.warp`
        takes: ``warp(moved, field, axes)`` resamples ``moved`` onto
        ``reference``.
    inverse_field : torch.Tensor or None
        Same layout, ``reference(p + inverse_field[p]) ≈ moved(p)``.  ``None``
        with ``optical_flow``: BART computes no inverse there
        (``estmotion.c:275``).
    """
    _check_same_shape(reference, moved, "reference and moved")
    ndim = reference.ndim
    if ndim > _MOTION_DIM:
        raise ValueError(f"at most {_MOTION_DIM} axes, got {ndim}")
    axes = _normalize_axes(axes, ndim)
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
        return _from_bart_order(f, axes).movedim(0, -1).contiguous()

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
        Shift as a fraction of the axis length rather than in voxels (``-f``).

    Returns
    -------
    torch.Tensor
        Real, one shift per entry of ``axes`` in that order:
        ``a ≈ torch.roll(b, shifts, axes)`` for whole-voxel shifts.
    """
    _check_same_shape(a, b, "a and b")
    axes = _normalize_axes(axes, a.ndim)
    shifts = dispatch("estshift", [a, b], None, _pos=[axes_flags(axes, a.ndim)], f=fov_units)
    return _from_bart_order(shifts.reshape(-1).real, axes).contiguous()
