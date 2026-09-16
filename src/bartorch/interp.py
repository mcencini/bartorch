"""Interpolation, warps, affine resampling and field-of-view shifts on a grid."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["affine_transform", "fovshift", "interpolate", "warp"]

#: BART's MOTION_DIM (= ITER_DIM, misc/mri.h), where a coordinate or
#: displacement field keeps its components.
_MOTION_DIM = 8

_ORDERS = (0, 1, 3)


def _normalize_axes(axes: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    axes = (axes,) if isinstance(axes, int) else tuple(int(a) for a in axes)
    out = []
    for a in axes:
        if not -ndim <= a < ndim:
            raise ValueError(f"axis {a} is out of range for {ndim} axes")
        out.append(a % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {axes}")
    return tuple(out)


def _order_flags(order: int) -> dict:
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {_ORDERS}, got {order}")
    return {"N": order == 0, "C": order == 3}


def _field(field, ndim: int, axes: tuple[int, ...], what: str) -> tuple[torch.Tensor, tuple]:
    """A ``(..., len(axes))`` field as BART lays it out, and its grid shape.

    The grid shape is right-aligned against ``ndim`` axes.  Components are
    reordered from the order of ``axes`` to increasing BART dim, the order
    ``md_positions`` and ``interpolate2`` walk the flagged dims in
    (motion/interpolate.c:69, 257), and moved onto MOTION_DIM.
    """
    field = torch.as_tensor(field)
    if field.is_complex():
        field = field.real
    m = len(axes)
    if field.ndim < 1 or field.shape[-1] != m:
        raise ValueError(f"{what} must end in an axis of {m} components, got {tuple(field.shape)}")
    grid = tuple(field.shape[:-1])
    if len(grid) > ndim:
        raise ValueError(f"{what} has more grid axes than the input's {ndim}")
    grid = (1,) * (ndim - len(grid)) + grid
    perm = sorted(range(m), key=lambda i: -axes[i])
    field = field[..., perm].reshape(*grid, m).to(torch.complex64)
    if ndim <= _MOTION_DIM:
        field = field.movedim(-1, 0).reshape(m, *([1] * (_MOTION_DIM - ndim)), *grid)
    else:
        at = ndim - 1 - _MOTION_DIM
        field = field.movedim(-1, at).reshape(*grid[:at], m, *grid[at + 1 :])
    return field.contiguous(), grid


def _check_motion_dim(input: torch.Tensor, axes: tuple[int, ...]) -> None:
    ndim = input.ndim
    if ndim > _MOTION_DIM:
        at = ndim - 1 - _MOTION_DIM
        if input.shape[at] != 1 or at in axes:
            raise ValueError(
                f"axis {at} is BART's MOTION_DIM, which carries the field's components: "
                "it must be of size one and not interpolated"
            )


@curated("interpolate")
def interpolate(
    input: torch.Tensor,
    coord: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    order: int = 1,
) -> torch.Tensor:
    """Sample ``input`` at the voxel positions ``coord`` along ``axes``.

    ``out[..., q, ...] = sum_j input[..., j, ...] w(coord[q, i] - j_i)`` over
    the ``axes``, where ``w`` is the kernel of ``order``.  Positions are array
    indices: ``coord = 2.0`` is the sample ``input[2]``.  The grid is
    zero outside ``[0, n - 1]``, so a position within one kernel radius of
    the edge is weighted against zeros.

    Parameters
    ----------
    input : torch.Tensor
    coord : torch.Tensor
        Real positions of shape ``(..., len(axes))``; component ``i`` of the
        last axis is the position along ``axes[i]``.  The leading shape is
        aligned right against ``input.shape``: along each of ``axes`` it is
        the output size, along every other axis it is one (the positions are
        shared) or the input's size.  Along an axis outside the last three
        (``bart interpolate -x`` sizes only BART dims 0-2) the output size
        must equal the input's.
    axes : int or tuple of int
        Axes interpolated over.
    order : {0, 1, 3}
        Nearest neighbour (rounding half up), linear, or Keys cubic
        with ``a = -1/2``.

    Returns
    -------
    torch.Tensor
        ``input.shape`` with each of ``axes`` replaced by ``coord``'s size
        there.

    Examples
    --------
    >>> line = interpolate(image, torch.tensor([[1.5, 2.0]]), axes=(-2, -1))  # (1, 1)
    """
    ndim = input.ndim
    axes = _normalize_axes(axes, ndim)
    _check_motion_dim(input, axes)
    field, grid = _field(coord, ndim, axes, "coord")
    out_shape = list(input.shape)
    for ax in range(ndim):
        size, have = grid[ax], input.shape[ax]
        if ax in axes:
            if ndim - 1 - ax > 2 and size != have:
                raise ValueError(
                    f"coord has size {size} along axis {ax}, outside the last three: "
                    f"it must equal the input's {have}"
                )
            out_shape[ax] = size
        elif size not in (1, have):
            raise ValueError(
                f"coord has size {size} along axis {ax}, which is not interpolated: "
                f"it must be one or the input's {have}"
            )
    x = tuple(out_shape[ndim - 1 - b] if b < ndim else 1 for b in range(3))
    out = dispatch(
        "interpolate",
        [input, field],
        None,
        _pos=[axes_flags(axes, ndim)],
        x=x,
        **_order_flags(order),
    )
    return out.reshape(out_shape)


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
    axes = _normalize_axes(axes, ndim)
    if set(axes) != set(range(ndim - len(axes), ndim)):
        raise ValueError(f"warp axes must be the last {len(axes)} axes, got {axes}")
    _check_motion_dim(input, axes)
    field, grid = _field(displacement, ndim, axes, "displacement")
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
        **_order_flags(order),
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
    axes = _normalize_axes(axes, ndim)
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
        **_order_flags(order),
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
