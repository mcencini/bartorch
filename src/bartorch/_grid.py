"""What the operations on a grid share: axis checks, and BART's motion layout.

:mod:`bartorch.interp` and :mod:`bartorch.tools.process` both resample on a
grid and both reach BART's ``interpolate`` through the same conventions, so the
conventions live here rather than in either of them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: BART's MOTION_DIM (= ITER_DIM, misc/mri.h), where a coordinate or
#: displacement field keeps its components.
MOTION_DIM = 8

#: The interpolation orders BART's ``interpolate`` implements.
ORDERS = (0, 1, 3)


def same_shape(a: torch.Tensor, b: torch.Tensor, names: str) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"{names} must have the same shape, got {tuple(a.shape)} and {tuple(b.shape)}"
        )


def normalize_axes(axes: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    axes = (axes,) if isinstance(axes, int) else tuple(int(a) for a in axes)
    out = []
    for a in axes:
        if not -ndim <= a < ndim:
            raise ValueError(f"axis {a} is out of range for {ndim} axes")
        out.append(a % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {axes}")
    return tuple(out)


def order_flags(order: int) -> dict:
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order}")
    return {"N": order == 0, "C": order == 3}


def from_bart_order(components: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    """Reorder a leading component axis from BART's (descending C axis) to that of ``axes``."""
    descending = sorted(axes, reverse=True)
    return components[[descending.index(a) for a in axes]]


def motion_field(field, ndim: int, axes: tuple[int, ...], what: str) -> tuple[torch.Tensor, tuple]:
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
    if ndim <= MOTION_DIM:
        field = field.movedim(-1, 0).reshape(m, *([1] * (MOTION_DIM - ndim)), *grid)
    else:
        at = ndim - 1 - MOTION_DIM
        field = field.movedim(-1, at).reshape(*grid[:at], m, *grid[at + 1 :])
    return field.contiguous(), grid


def check_motion_dim(input: torch.Tensor, axes: tuple[int, ...]) -> None:
    ndim = input.ndim
    if ndim > MOTION_DIM:
        at = ndim - 1 - MOTION_DIM
        if input.shape[at] != 1 or at in axes:
            raise ValueError(
                f"axis {at} is BART's MOTION_DIM, which carries the field's components: "
                "it must be of size one and not interpolated"
            )
