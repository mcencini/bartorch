"""Image-quality metrics."""

from __future__ import annotations

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["mse", "nrmse", "psnr", "roi_stat", "ssim"]

_STATS = {"count": "C", "sum": "S", "mean": "M", "std": "D", "energy": "E", "variance": "V"}


def _check_same_shape(reference: torch.Tensor, input: torch.Tensor) -> None:
    if tuple(reference.shape) != tuple(input.shape):
        raise ValueError(
            "reference and input must have the same shape, "
            f"got {tuple(reference.shape)} and {tuple(input.shape)}"
        )


@curated("nrmse")
def nrmse(reference: torch.Tensor, input: torch.Tensor, *, scaled: bool = False) -> float:
    """Normalized root mean square error ``‖input - reference‖ / ‖reference‖``.

    Parameters
    ----------
    scaled : bool
        First scale ``reference`` by the complex least-squares factor
        ``Σ conj(reference)·input / ‖reference‖²``.
    """
    _check_same_shape(reference, input)
    return float(dispatch("nrmse", [reference, input], False, s=scaled, scientific=True))


def _measure(option: str, reference: torch.Tensor, input: torch.Tensor) -> float:
    _check_same_shape(reference, input)
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
