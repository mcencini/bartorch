"""Denoisers, for plug-and-play priors."""

from __future__ import annotations

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["nlmeans", "rof", "tgv"]

# BART's three spatial dims, the last three axes of a C-order tensor.
_FFT_FLAGS = 7
# `tgv` keeps its auxiliary variables along BART's last dim.
_MAX_NDIM = 15
# `--tvscales` holds at most this many values.
_MAX_SCALES = 5


def _axes_tuple(axes: int | tuple[int, ...]) -> tuple[int, ...]:
    return (axes,) if isinstance(axes, int) else tuple(axes)


def _check_ndim(input: torch.Tensor, command: str) -> None:
    if input.ndim > _MAX_NDIM:
        raise ValueError(f"bart {command} takes at most {_MAX_NDIM} axes, got {input.ndim}")


def _scales(values, axes, ndim: int, option: str) -> tuple[float, ...] | None:
    """Per-axis scales, reordered from ``axes`` to BART's bit order."""
    if values is None:
        return None
    axes = _axes_tuple(axes)
    values = tuple(float(v) for v in values)
    if len(values) != len(axes):
        raise ValueError(f"{option} needs one value per axis in {axes}, got {len(values)}")
    if len(values) > _MAX_SCALES:
        raise ValueError(f"{option} takes at most {_MAX_SCALES} axes")
    if 0.0 in values:
        # BART drops trailing zeros and then fails an assertion on the count.
        raise ValueError(f"{option} values must be nonzero")
    bits = [axes_flags(a, ndim) for a in axes]
    return tuple(v for _, v in sorted(zip(bits, values, strict=True)))


def _image(out: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """The denoised image: the first block of an output that appends auxiliary variables."""
    return out.reshape(-1)[: input.numel()].reshape(input.shape)


@curated("rof")
def rof(input: torch.Tensor, lamda: float, axes: int | tuple[int, ...]) -> torch.Tensor:
    """Total-variation (Rudin-Osher-Fatemi) denoising along ``axes``.

    Approximates ``argmin_x 0.5 ||x - y||_2^2 + lamda * sum_r ||(D x)_r||_2``, where
    ``D`` stacks periodic first differences along ``axes`` and the inner norm is over
    the derivative directions (isotropic TV of complex values).  BART runs 50 ADMM
    iterations with ``rho = 0.1``, so a large ``lamda`` does not reach the constant
    minimizer; the mean of ``input`` is preserved.

    Parameters
    ----------
    lamda : float
        Weight of the TV term relative to the halved squared data error, in the
        units of ``input``'s values.
    axes : int or tuple of int
        Axes along which differences are taken.
    """
    return dispatch("rof", [input], None, _pos=[float(lamda), axes_flags(axes, input.ndim)])


@curated("tgv")
def tgv(
    input: torch.Tensor,
    lamda: float,
    axes: int | tuple[int, ...],
    *,
    alpha: tuple[float, float] | None = None,
    tvscales: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Second-order total generalized variation denoising along ``axes``.

    Approximates ``argmin_{x,z} 0.5 ||x - y||_2^2 + lamda * (alpha1 ||D x - z||_1 +
    alpha0 ||E z||_1)``, where ``D`` is the periodic gradient along ``axes``, ``z`` a
    vector field and ``E z = (D z + (D z)^T) / 2`` its symmetrized gradient; each
    ``||.||_1`` sums over voxels an l2 norm over derivative directions.  BART runs 100
    ADMM iterations with ``rho = 0.1``.  Only ``x`` is returned; the mean of ``input``
    is preserved.

    Parameters
    ----------
    input : torch.Tensor
        At most 15 axes.
    alpha : (float, float), optional
        The pair ``(alpha1, alpha0)``; BART's default is ``(1, sqrt(3))``.
    tvscales : tuple of float, optional
        Nonzero weight on the derivative along each axis in ``axes``, in the same
        order, applied in both ``D`` and ``E``.
    """
    _check_ndim(input, "tgv")
    out = dispatch(
        "tgv",
        [input],
        None,
        _pos=[float(lamda), axes_flags(axes, input.ndim)],
        alpha=None if alpha is None else tuple(float(a) for a in alpha),
        tvscales=_scales(tvscales, axes, input.ndim, "tvscales"),
    )
    return _image(out, input)


@curated("nlmeans")
def nlmeans(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    patch_length: int | None = None,
    patch_distance: int | None = None,
    h: float | None = None,
    a: float | None = None,
) -> torch.Tensor:
    """Non-local means filter along ``axes``.

    Each voxel becomes ``sum_j w_ij y_j / sum_j w_ij`` over the ``j`` within
    ``patch_distance`` of it along every axis in ``axes``, with ``w_ij = exp(-sum_o
    g_o |y_(i+o) - y_(j+o)|^2 / (2 h^2 ||g||_2))``: ``o`` runs over a patch of
    ``patch_length`` per axis and ``g`` is a Gaussian of standard deviation ``a`` over
    it.  The image is reflected at its edges.

    Parameters
    ----------
    patch_length : int, optional
        Odd patch length per axis; BART's default is 5.
    patch_distance : int, optional
        Search radius per axis; BART's default is 5.  Zero returns ``input``.
    h : float, optional
        Filter strength in the units of ``input``'s values; BART's default
        is 0.04.  A large ``h`` makes the filter a box mean.
    a : float, optional
        Standard deviation of ``g``, in voxels; BART's default is
        ``(patch_length - 1) / 4``.
    """
    if patch_length is not None and int(patch_length) % 2 != 1:
        raise ValueError(f"patch_length must be odd, got {patch_length}")
    return dispatch(
        "nlmeans",
        [input],
        None,
        _pos=[axes_flags(axes, input.ndim)],
        patch_length=None if patch_length is None else int(patch_length),
        patch_dist=None if patch_distance is None else int(patch_distance),
        H=None if h is None else float(h),
        a=None if a is None else float(a),
    )
