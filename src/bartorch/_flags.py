"""C-order axis indices to BART dimensions and bitmasks."""

from __future__ import annotations

__all__: list[str] = []


def _axes_to_dims(axes: int | tuple[int, ...] | list[int], ndim: int | None) -> list[int]:
    """BART dimensions of C-order ``axes`` of an ``ndim``-axis array.

    C-order axis ``a`` is BART dimension ``ndim - 1 - a``, and a negative axis
    ``a`` is dimension ``-a - 1`` whatever ``ndim`` is.  With ``ndim`` None --
    a command that reads no array to count from -- only negative axes are
    accepted.

    Raises
    ------
    ValueError
        An axis is out of range or repeated, ``ndim < 1``, or ``ndim`` is None
        and an axis is not negative.

    Examples
    --------
    >>> _axes_to_dims((-1, -2), ndim=3)
    [0, 1]
    >>> _axes_to_dims(0, ndim=2)
    [1]
    >>> _axes_to_dims(-3, ndim=None)
    [2]
    """
    if ndim is not None and ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")

    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    dims: list[int] = []
    for orig in axes:
        if isinstance(orig, bool) or not isinstance(orig, int):
            raise TypeError(f"an axis is an int, not {orig!r}")
        if ndim is None:
            if orig >= 0:
                raise ValueError(
                    f"axis {orig} needs an array to count from, and there is none; "
                    "count from the last axis instead, with a negative index"
                )
            dims.append(-orig - 1)
            continue
        a = orig + ndim if orig < 0 else orig
        if a < 0 or a >= ndim:
            raise ValueError(
                f"axis {orig} out of range for ndim={ndim} (valid range: [{-ndim}, {ndim - 1}])"
            )
        dims.append(ndim - 1 - a)

    if len(dims) != len(set(dims)):
        raise ValueError("duplicate axis indices are not allowed")
    return dims


def _axes_to_flags(axes: int | tuple[int, ...] | list[int], ndim: int | None) -> int:
    """BART bitmask of C-order ``axes`` of an ``ndim``-axis array; see :func:`_axes_to_dims`.

    Examples
    --------
    >>> _axes_to_flags((-1, -2), ndim=3)
    3
    >>> _axes_to_flags(0, ndim=2)
    2
    """
    return sum(1 << d for d in _axes_to_dims(axes, ndim))


def _indices_to_flags(indices: int | tuple[int, ...] | list[int]) -> int:
    """BART bitmask of a set of indices -- channels, parameter maps -- that are not axes.

    Examples
    --------
    >>> _indices_to_flags((0, 2))
    5
    """
    indices = (indices,) if isinstance(indices, int) else tuple(indices)
    flags = 0
    for i in indices:
        if isinstance(i, bool) or not isinstance(i, int) or i < 0:
            raise ValueError(f"an index is a non-negative int, not {i!r}")
        if flags & (1 << i):
            raise ValueError(f"index {i} is repeated")
        flags |= 1 << i
    return flags
