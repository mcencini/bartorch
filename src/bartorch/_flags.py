"""C-order axis indices to BART bitmasks."""

from __future__ import annotations

__all__: list[str] = []


def _axes_to_flags(
    axes: int | tuple[int, ...] | list[int],
    ndim: int,
) -> int:
    """BART bitmask of C-order ``axes`` of an ``ndim``-axis array.

    C-order axis ``a`` is BART axis ``ndim - 1 - a``; negative axes count from the end.

    Raises
    ------
    ValueError
        An axis is out of range or repeated, or ``ndim < 1``.

    Examples
    --------
    >>> _axes_to_flags((-1, -2), ndim=3)
    3
    >>> _axes_to_flags(0, ndim=2)
    2
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")

    if isinstance(axes, int):
        axes = (axes,)
    else:
        axes = tuple(axes)

    normalised: list[int] = []
    for orig in axes:
        a = orig + ndim if orig < 0 else orig
        if a < 0 or a >= ndim:
            raise ValueError(
                f"axis {orig} out of range for ndim={ndim} (valid range: [{-ndim}, {ndim - 1}])"
            )
        normalised.append(a)

    if len(normalised) != len(set(normalised)):
        raise ValueError("duplicate axis indices are not allowed")

    flags = 0
    for a in normalised:
        bart_axis = ndim - 1 - a
        flags |= 1 << bart_axis

    return flags
