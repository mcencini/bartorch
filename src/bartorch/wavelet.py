"""Wavelet transforms."""

from __future__ import annotations

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["fwt", "iwt"]

#: BART ``wavelet`` option and filter length per wavelet (``src/wavelet/wavelet.c``).
_FILTERS = {"haar": ("H", 2), "dau2": ("D", 4), "cdf44": ("C", 10)}
_WAVELETS = (*_FILTERS, "cdf97")

#: Coarsest scale BART's ``wavelet`` command decomposes to, per axis.
_MIN_SIZE = 16


def _axes(axes: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Normalized, sorted, distinct C-order axes."""
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    out = []
    for ax in axes:
        if not -ndim <= ax < ndim:
            raise ValueError(f"axis {ax} is out of range for {ndim} dimensions")
        out.append(ax % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {axes}")
    if not out:
        raise ValueError("no axes to transform")
    return tuple(sorted(out))


def _check_wavelet(wavelet: str) -> None:
    if wavelet not in _WAVELETS:
        raise ValueError(f"wavelet must be one of {_WAVELETS}, got {wavelet!r}")


def _check_sizes(shape: tuple[int, ...], axes: tuple[int, ...], what: str) -> None:
    # BART asserts on a smaller axis (wavelet_check_dims, src/wavelet/wavelet.c).
    small = [shape[ax] for ax in axes if shape[ax] < _MIN_SIZE]
    if small:
        raise ValueError(f"{what} axes {axes} of shape {shape}: every one needs >= {_MIN_SIZE}")


def _num_coeffs(sizes: list[int], flen: int) -> int:
    """Length of the coefficient axis for ``sizes``: ``wavelet_coeffs2`` in BART."""

    def bands(dims, active):
        return [(n + flen - 1) // 2 if a else n for n, a in zip(dims, active, strict=True)]

    def prod(values):
        out = 1
        for v in values:
            out *= v
        return out

    dims, active, total = list(sizes), [True] * len(sizes), 0
    while any(active):
        wdims = bands(dims, active)
        total += prod(wdims) * (2 ** sum(active) - 1)
        active = [a and n >= _MIN_SIZE for n, a in zip(wdims, active, strict=True)]
        dims = wdims
    return total + prod(dims)


@curated("wavelet", "cdf97")
def fwt(input: torch.Tensor, axes: int | tuple[int, ...], *, wavelet: str = "dau2") -> torch.Tensor:
    """Forward multi-level discrete wavelet transform along ``axes``.

    Parameters
    ----------
    input : torch.Tensor
        Every transformed axis needs at least 16 samples, the coarsest scale.
    axes : int or tuple of int
        Axes to transform, as indices into ``input.shape``.
    wavelet : {"haar", "dau2", "cdf44", "cdf97"}
        ``"haar"``, ``"dau2"`` and ``"cdf44"`` are, along one axis,
        PyWavelets' ``wavedec`` with ``"haar"``, ``"db2"`` and ``"bior4.4"``
        in ``mode="symmetric"``, concatenated coarse first.  Each level keeps
        ``(n + taps - 1) // 2`` coefficients per band, so there are more
        coefficients than samples unless the filter has two taps and every
        level's length is even; levels continue while that half-length is at
        least 16.  ``"cdf97"`` is CDF 9/7 by lifting, in place, coarse band
        first, levels continuing while every transformed axis is longer than
        32; one level of an even length is PyWavelets' ``"bior4.4"`` in
        ``mode="periodization"`` with the detail band negated.

    Returns
    -------
    torch.Tensor
        For ``"cdf97"``, the shape of ``input``.  Otherwise the coefficients
        of all transformed axes are flattened into the last of ``axes``, and
        the other transformed axes have size one.
    """
    _check_wavelet(wavelet)
    shape = tuple(input.shape)
    ax = _axes(axes, input.ndim)
    flags = axes_flags(ax, input.ndim)
    if wavelet == "cdf97":
        return dispatch("cdf97", [input], None, _pos=[flags]).reshape(shape)
    _check_sizes(shape, ax, "transformed")
    option, flen = _FILTERS[wavelet]
    out = list(shape)
    for a in ax:
        out[a] = 1
    out[ax[-1]] = _num_coeffs([shape[a] for a in ax], flen)
    return dispatch("wavelet", [input], None, _pos=[flags], **{option: True}).reshape(out)


@curated("wavelet", "cdf97")
def iwt(
    input: torch.Tensor,
    oshape: tuple[int, ...],
    axes: int | tuple[int, ...],
    *,
    wavelet: str = "dau2",
) -> torch.Tensor:
    """Wavelet synthesis along ``axes``: coefficients from :func:`fwt` back to samples.

    ``bart wavelet -a`` runs the synthesis filter bank, which inverts
    :func:`fwt` (``iwt(fwt(x)) == x``) for every wavelet.  It is a left
    inverse, not the adjoint, although BART's wavelet linop uses it as its
    adjoint: synthesis ignores the reflected samples the analysis read
    (``wavelet_up3``, ``src/wavelet/wavelet.c``).  The two coincide only for
    ``"haar"`` when every level's length is even, where the transform is
    orthogonal.  ``"cdf97"`` runs ``bart cdf97 -i``, the exact inverse of a
    biorthogonal transform that is not orthogonal.

    Parameters
    ----------
    input : torch.Tensor
        Coefficients, laid out as :func:`fwt` returns them.
    oshape : tuple of int
        Shape of the result, C order, one entry per axis of ``input``; the
        axes not in ``axes`` must match ``input``.
    axes : int or tuple of int
        Transformed axes, as indices into ``oshape``.
    wavelet : {"haar", "dau2", "cdf44", "cdf97"}
        As for :func:`fwt`.
    """
    _check_wavelet(wavelet)
    oshape = tuple(int(n) for n in oshape)
    ndim = len(oshape)
    if input.ndim > ndim:
        raise ValueError(f"input of shape {tuple(input.shape)} has more axes than {oshape}")
    input = input.reshape((1,) * (ndim - input.ndim) + tuple(input.shape))
    ax = _axes(axes, ndim)
    flags = axes_flags(ax, ndim)
    if wavelet == "cdf97":
        if tuple(input.shape) != oshape:
            raise ValueError(f"cdf97 keeps the shape: input {tuple(input.shape)}, oshape {oshape}")
        return dispatch("cdf97", [input], None, _pos=[flags], i=True).reshape(oshape)
    _check_sizes(oshape, ax, "output")
    option, flen = _FILTERS[wavelet]
    expect = list(oshape)
    for a in ax:
        expect[a] = 1
    expect[ax[-1]] = _num_coeffs([oshape[a] for a in ax], flen)
    if list(input.shape) != expect:
        raise ValueError(
            f"{wavelet} coefficients of {oshape} along {ax} have shape {tuple(expect)}, "
            f"got {tuple(input.shape)}"
        )
    # The output sizes of the transformed axes, in BART's ascending dim order.
    sizes = [oshape[a] for a in reversed(ax)]
    return dispatch(
        "wavelet", [input], None, _pos=[flags, *sizes], a=True, **{option: True}
    ).reshape(oshape)
