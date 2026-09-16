"""Fourier transforms along C-order axes."""

from __future__ import annotations

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["fft", "fftmod", "fftshift", "ifft", "nufft", "nufft_adjoint"]


@curated("fft")
def fft(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    inverse: bool = False,
    unitary: bool = False,
    uncentred: bool = False,
) -> torch.Tensor:
    """Fourier transform along ``axes``, centred by default.

    Parameters
    ----------
    input : torch.Tensor
    axes : int or tuple of int
        Axes to transform, as indices into ``input.shape``.
    inverse : bool
        Transform with the positive exponent.
    unitary : bool
        Scale by one over the square root of the transformed size;
        otherwise unnormalized.
    uncentred : bool
        Keep the zero frequency at index zero rather than at ``n // 2``.

    Examples
    --------
    >>> spectrum = fft(image, axes=(-2, -1), unitary=True)
    """
    return dispatch(
        "fft",
        [input],
        None,
        _pos=[axes_flags(axes, input.ndim)],
        i=inverse,
        u=unitary,
        n=uncentred,
    )


def ifft(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    unitary: bool = False,
    uncentred: bool = False,
) -> torch.Tensor:
    """Inverse Fourier transform along ``axes``: :func:`fft` with ``inverse=True``."""
    return fft(input, axes, inverse=True, unitary=unitary, uncentred=uncentred)


@curated("fftmod")
def fftmod(
    input: torch.Tensor, axes: int | tuple[int, ...], *, inverse: bool = False
) -> torch.Tensor:
    """Multiply by the alternating phase that centres an uncentred transform, along ``axes``."""
    return dispatch("fftmod", [input], None, _pos=[axes_flags(axes, input.ndim)], i=inverse)


@curated("fftshift")
def fftshift(
    input: torch.Tensor, axes: int | tuple[int, ...], *, inverse: bool = False
) -> torch.Tensor:
    """Move the zero frequency to the middle along ``axes``, or back with ``inverse``."""
    return dispatch("fftshift", [input], None, _pos=[axes_flags(axes, input.ndim)], i=inverse)


@curated("nufft")
def nufft(
    input: torch.Tensor,
    traj: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    **extra,
) -> torch.Tensor:
    """Non-uniform Fourier transform of an image to samples along ``traj``.

    Computed by FINUFFT, with a negative exponent and scaled by one over the
    square root of the image's voxel count.

    Parameters
    ----------
    input : torch.Tensor
        Image, C order ``(..., z, y, x)``; ``z`` is one for a two-dimensional
        trajectory.
    traj : torch.Tensor
        Trajectory ``(..., samples, 3)`` in grid units, ``kx, ky, kz``, as
        :func:`bartorch.tools.traj` produces.
    weights : torch.Tensor, optional
        Diagonal in k-space applied to the samples.
    basis : torch.Tensor, optional
        Temporal subspace basis, coefficients then frames, in BART's axis
        order: ``(coeffs, frames, 1, 1, 1, 1, 1)``.
    **extra
        Further ``bart nufft`` options, by name.
    """
    flags: dict = dict(extra)
    if weights is not None:
        flags["p"] = weights
    if basis is not None:
        flags["B"] = basis
    return dispatch("nufft", [traj, input], None, **flags)


@curated("nufft")
def nufft_adjoint(
    input: torch.Tensor,
    traj: torch.Tensor,
    image_shape: tuple[int, ...] | None = None,
    *,
    weights: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    **extra,
) -> torch.Tensor:
    """Adjoint of :func:`nufft`: samples along ``traj`` back to an image.

    Parameters
    ----------
    input : torch.Tensor
        Samples, as :func:`nufft` returns them.
    traj : torch.Tensor
        Trajectory ``(..., samples, 3)`` in grid units, ``kx, ky, kz``, as
        :func:`bartorch.tools.traj` produces.
    image_shape : tuple of int, optional
        Spatial shape of the image, C order ``(z, y, x)`` or ``(y, x)``.  By
        default BART estimates it from the trajectory.
    weights : torch.Tensor, optional
        Diagonal in k-space applied to the samples; its conjugate is applied
        here.
    basis : torch.Tensor, optional
        Temporal subspace basis, coefficients then frames, in BART's axis
        order: ``(coeffs, frames, 1, 1, 1, 1, 1)``.
    **extra
        Further ``bart nufft`` options, by name.
    """
    flags: dict = dict(extra)
    flags["a"] = True
    if image_shape is not None:
        spatial = [int(n) for n in reversed(tuple(image_shape))]
        flags["d"] = tuple((spatial + [1, 1, 1])[:3])
    if weights is not None:
        flags["p"] = weights
    if basis is not None:
        flags["B"] = basis
    return dispatch("nufft", [traj, input], None, **flags)
