"""Array utilities along C-order axes."""

from __future__ import annotations

import math

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = [
    "casorati",
    "circshift",
    "conv",
    "flip",
    "kernels_to_maps",
    "maps_to_kernels",
    "median_filter",
    "mip",
    "moving_average",
    "normalize",
    "resize",
    "rss",
    "unwrap",
    "window",
]


@curated("flip")
def flip(input: torch.Tensor, axes: int | tuple[int, ...]) -> torch.Tensor:
    """Reverse ``input`` along ``axes``."""
    return dispatch("flip", [input], None, _pos=[axes_flags(axes, input.ndim)])


@curated("rss")
def rss(input: torch.Tensor, axes: int | tuple[int, ...], *, keepdim: bool = False) -> torch.Tensor:
    """Root of the sum of squared magnitudes over ``axes``.

    The reduced axes are removed from the shape, or kept as size one with
    ``keepdim``.
    """
    flags = axes_flags(axes, input.ndim)
    reduced = {input.ndim - 1 - b for b in range(input.ndim) if flags >> b & 1}
    shape = [
        1 if axis in reduced else n
        for axis, n in enumerate(input.shape)
        if keepdim or axis not in reduced
    ]
    return dispatch("rss", [input], None, _pos=[flags]).reshape(shape)


def _resize_centre(x: torch.Tensor, spatial) -> torch.Tensor:
    """Crop or zero-pad the trailing axes about the index ``n // 2``.

    The offset between two sizes is the difference of their halves, which
    differs from half their difference when one size is odd; this matches
    ``md_resize_center`` and so the centre of a centred FFT.
    """
    spatial = tuple(spatial)
    out = x
    for axis, want in zip(range(-len(spatial), 0), spatial, strict=True):
        have = out.shape[axis]
        if have == want:
            continue
        offset = abs(want // 2 - have // 2)
        if want < have:
            out = out.narrow(axis, offset, want)
        else:
            shape = list(out.shape)
            shape[axis] = want
            padded = torch.zeros(shape, dtype=out.dtype, device=out.device)
            padded.narrow(axis, offset, have).copy_(out)
            out = padded
    return out


def maps_to_kernels(
    maps: torch.Tensor, size: int | tuple[int, ...], ndim: int | None = None
) -> torch.Tensor:
    """Coil sensitivities as k-space kernels: the centre of each map's unitary spectrum.

    What :class:`bartorch.linop.NoncartesianSense` takes with ``kernels=True``.  Content
    outside the kernel's band is lost; :func:`kernels_to_maps` gives the maps
    the kernels stand for, to measure that loss.

    Parameters
    ----------
    maps : torch.Tensor
        Sensitivities ``([sets,] coils, *spatial)``.
    size : int or tuple of int
        Kernel size per spatial axis, or one size for all.
    ndim : int, optional
        Spatial axes, for a single ``size`` over a bank with sets.  By default
        the length of ``size``, or every axis after the first.

    Returns
    -------
    torch.Tensor
        Kernels ``([sets,] coils, *size)``.
    """
    from bartorch.fourier import fft

    if ndim is None:
        ndim = maps.ndim - 1 if isinstance(size, int) else len(tuple(size))
    spatial = tuple(maps.shape[maps.ndim - ndim :])
    size = (size,) * len(spatial) if isinstance(size, int) else tuple(size)
    if len(size) != len(spatial):
        raise ValueError(f"kernel size {size} does not match the map's {spatial} spatial axes")
    axes = tuple(range(-len(spatial), 0))
    return _resize_centre(fft(maps, axes=axes, unitary=True), size)


def kernels_to_maps(kernels: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
    """The maps a kernel bank stands for: zero-padded to ``spatial`` and transformed back."""
    from bartorch.fourier import fft

    spatial = tuple(spatial)
    axes = tuple(range(-len(spatial), 0))
    return fft(_resize_centre(kernels, spatial), axes=axes, unitary=True, inverse=True)


def _bart_dim(axis: int, ndim: int) -> int:
    """BART dimension index of the C-order ``axis`` of an ``ndim``-dimensional tensor."""
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of range for {ndim} dimensions")
    return ndim - 1 - (axis % ndim)


def _as_tuple(value) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


@curated("resize")
def resize(input: torch.Tensor, oshape: tuple[int, ...], *, anchor: str = "center") -> torch.Tensor:
    """Crop or zero-pad ``input`` to ``oshape``.

    Parameters
    ----------
    oshape : tuple of int
        Full output shape, one size per axis of ``input``.
    anchor : {"center", "front", "start"}
        What is kept in place.  ``"center"`` (``-c``) keeps index ``n // 2``
        at ``m // 2``; ``"front"`` (``-f``) keeps the last element and crops or
        pads at the beginning; ``"start"`` keeps the first element and crops or
        pads at the end.
    """
    oshape = tuple(int(n) for n in oshape)
    if len(oshape) != input.ndim:
        raise ValueError(f"oshape {oshape} does not have the {input.ndim} axes of the input")
    if any(n < 1 for n in oshape):
        raise ValueError(f"oshape {oshape} has a size below one")
    if anchor not in ("center", "front", "start"):
        raise ValueError(f"anchor must be 'center', 'front' or 'start', not {anchor!r}")
    ndim = input.ndim
    pos: list[int] = []
    for axis, (have, want) in enumerate(zip(input.shape, oshape, strict=True)):
        if have != want:
            pos += [ndim - 1 - axis, want]
    if not pos:  # BART takes at least one pair: restate the last axis's size.
        pos = [0, oshape[-1]]
    out = dispatch("resize", [input], None, _pos=pos, c=anchor == "center", f=anchor == "front")
    return out.reshape(oshape)


@curated("circshift")
def circshift(
    input: torch.Tensor, shifts: int | tuple[int, ...], axes: int | tuple[int, ...]
) -> torch.Tensor:
    """Cyclic shift along ``axes``: element ``i`` moves to ``(i + shift) % n``, as torch.roll."""
    shifts, axes = _as_tuple(shifts), _as_tuple(axes)
    if len(shifts) != len(axes):
        raise ValueError(f"{len(shifts)} shifts for {len(axes)} axes")
    out = input
    for shift, axis in zip(shifts, axes, strict=True):
        dim = _bart_dim(axis, input.ndim)
        # BART walks a negative shift up one period at a time.
        shift = int(shift) % input.shape[axis] if input.shape[axis] else 0
        out = dispatch("circshift", [out], None, _pos=[dim, shift]).reshape(input.shape)
    return out


@curated("conv")
def conv(input: torch.Tensor, kernel: torch.Tensor, axes: int | tuple[int, ...]) -> torch.Tensor:
    """Cyclic convolution of ``input`` with ``kernel`` along ``axes``, of the input's shape.

    Kernel index ``(k - 1) // 2`` sits at zero shift, so an odd kernel is
    centred and an even one leans to the start:
    ``out[i] = sum_j kernel[j] * input[(i - j + (k - 1) // 2) % n]``.

    Parameters
    ----------
    kernel : torch.Tensor
        Aligned with ``input`` from the last axis.  Along ``axes`` any size;
        along the other axes one (shared) or the input's size (per slice).
    """
    if kernel.ndim > input.ndim:
        raise ValueError(f"kernel has {kernel.ndim} axes, more than the input's {input.ndim}")
    flags = axes_flags(axes, input.ndim)
    return dispatch("conv", [input, kernel], None, _pos=[flags]).reshape(input.shape)


@curated("window")
def window(input: torch.Tensor, axes: int | tuple[int, ...], *, hann: bool = False) -> torch.Tensor:
    """Multiply by a Hamming window along each of ``axes``, or a Hann window (``-H``).

    The window along an axis of size ``n`` is
    ``a - (1 - a) * cos(2 pi i / (n - 1))`` with ``a = 0.54`` (Hamming) or
    ``0.5`` (Hann), as :func:`numpy.hamming` and :func:`numpy.hanning`; a
    size-one axis is left alone.
    """
    flags = axes_flags(axes, input.ndim)
    return dispatch("window", [input], None, _pos=[flags], H=hann).reshape(input.shape)


def _filter_shape(input: torch.Tensor, axis: int, length: int) -> tuple[int, ...]:
    dim = _bart_dim(axis, input.ndim)
    axis = input.ndim - 1 - dim
    n = input.shape[axis]
    if not 1 <= length <= n:
        raise ValueError(f"filter length {length} must lie in [1, {n}] along axis {axis}")
    shape = list(input.shape)
    shape[axis] = n - length + 1
    return tuple(shape)


@curated("filter")
def median_filter(
    input: torch.Tensor, axis: int, length: int, *, geometric: bool = False
) -> torch.Tensor:
    """Median over each window of ``length`` consecutive elements along ``axis`` (``-m``, ``-l``).

    Only whole windows are taken: the axis shrinks from ``n`` to
    ``n - length + 1`` and ``out[i]`` is the median of ``input[i:i + length]``.

    Parameters
    ----------
    geometric : bool
        Take the geometric median of the complex values in the plane
        (``-G``), by ten Weiszfeld iterations from zero.  Otherwise the
        element of median magnitude, or the mean of the two for an even length.
    """
    shape = _filter_shape(input, axis, length)
    dim = _bart_dim(axis, input.ndim)
    return dispatch("filter", [input], None, m=dim, l=int(length), G=geometric).reshape(shape)


@curated("filter")
def moving_average(input: torch.Tensor, axis: int, length: int) -> torch.Tensor:
    """Mean over each window of ``length`` consecutive elements along ``axis`` (``-a``, ``-l``).

    Only whole windows are taken: the axis shrinks from ``n`` to
    ``n - length + 1`` and ``out[i]`` is the mean of ``input[i:i + length]``.
    """
    shape = _filter_shape(input, axis, length)
    dim = _bart_dim(axis, input.ndim)
    return dispatch("filter", [input], None, a=dim, l=int(length)).reshape(shape)


@curated("normalize")
def normalize(
    input: torch.Tensor, axes: int | tuple[int, ...], *, l1: bool = False
) -> torch.Tensor:
    """Divide by the l2 norm over ``axes``, or the l1 norm (``-b``), per index of the other axes."""
    flags = axes_flags(axes, input.ndim)
    return dispatch("normalize", [input], None, _pos=[flags], b=l1).reshape(input.shape)


@curated("mip")
def mip(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    minimum: bool = False,
    magnitude: bool = False,
) -> torch.Tensor:
    """Maximum (or minimum, ``-m``) intensity projection over ``axes``, which are kept as size one.

    Only real parts are compared and returned; the imaginary part of the
    output is zero.  The maximum starts from zero, so a projection whose real
    parts are all negative is zero.

    Parameters
    ----------
    magnitude : bool
        Project the magnitude instead of the real part (``-a``).
    """
    flags = axes_flags(axes, input.ndim)
    shape = [1 if flags >> (input.ndim - 1 - i) & 1 else n for i, n in enumerate(input.shape)]
    out = dispatch("mip", [input], None, _pos=[flags], m=minimum, a=magnitude)
    return out.reshape(shape)


@curated("unwrap")
def unwrap(input: torch.Tensor, axis: int, *, bound: float = math.pi) -> torch.Tensor:
    """Unwrap the real part along ``axis``: remove the jumps a period of ``2 * bound`` puts in it.

    A jump ``d`` between neighbours with ``|d| > bound`` is corrected by
    ``-sign(d) * ceil(|d| / bound) * bound``, which is ``2 * bound`` for any
    jump a wrapped signal can have (``|d| < 2 * bound``) and matches
    :func:`numpy.unwrap` with ``period=2 * bound`` there.  The imaginary part
    is left alone.

    Parameters
    ----------
    bound : float
        Half the period (``-b``).
    """
    dim = _bart_dim(axis, input.ndim)
    return dispatch("unwrap", [input], None, _pos=[dim], b=float(bound)).reshape(input.shape)


@curated("casorati")
def casorati(
    input: torch.Tensor, kernel_shape: int | tuple[int, ...], axes: int | tuple[int, ...]
) -> torch.Tensor:
    """Casorati matrix of the overlapping blocks of ``input``.

    A block spans ``kernel_shape`` along ``axes`` and the whole of every other
    axis; the blocks are all its positions, with no wrapping.

    Returns
    -------
    torch.Tensor
        Shape ``(block_size, n_blocks)``: BART's ``[blocks, block elements]``
        matrix read in C order.  Column ``s`` is block ``s`` flattened in C
        order, and blocks are numbered in C order of their start index.
    """
    kernel_shape, axes = _as_tuple(kernel_shape), _as_tuple(axes)
    if len(kernel_shape) != len(axes):
        raise ValueError(f"{len(kernel_shape)} kernel sizes for {len(axes)} axes")
    block = list(input.shape)
    pos: list[int] = []
    for size, axis in zip(kernel_shape, axes, strict=True):
        dim = _bart_dim(axis, input.ndim)
        ax = input.ndim - 1 - dim
        if not 1 <= int(size) <= input.shape[ax]:
            raise ValueError(
                f"kernel size {size} must lie in [1, {input.shape[ax]}] along axis {ax}"
            )
        block[ax] = int(size)
        pos += [dim, int(size)]
    n_blocks = math.prod(n - k + 1 for n, k in zip(input.shape, block, strict=True))
    out = dispatch("casorati", [input], None, _pos=pos)
    return out.reshape(math.prod(block), n_blocks)


_MORPHOLOGY = {"erosion": "e", "dilation": "d", "opening": "o", "closing": "c", "label": "l"}
