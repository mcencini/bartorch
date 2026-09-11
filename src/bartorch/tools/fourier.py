"""Transforms, on axes rather than on bitmasks.

BART selects axes with a bitmask over its own Fortran order.  Nothing above
this layer works that way, so every wrapper here takes an axis index into the
C-order shape -- negative indices included -- and converts it.
"""

from __future__ import annotations

import torch

from bartorch._operator import axes_flags
from bartorch.core.graph import dispatch
from bartorch.tools._call import curated

__all__ = ["fft", "fftmod", "fftshift", "ifft", "nufft"]


def _flags(axes, ndim: int) -> int:
    return axes_flags(axes, ndim)


@curated("fft")
def fft(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    inverse: bool = False,
    unitary: bool = False,
    uncentred: bool = False,
) -> torch.Tensor:
    """The Fourier transform along ``axes``.

    Parameters
    ----------
    input : torch.Tensor
        The array to transform, C order.
    axes : int or tuple of int
        Which axes to transform, as indices into ``input.shape``.  Negative
        indices count from the end.
    inverse : bool
        Transform the other way (``-i``).
    unitary : bool
        Scale by one over the square root of the transformed size, so that the
        transform preserves the norm (``-u``).
    uncentred : bool
        Leave the zero frequency at index zero rather than in the middle
        (``-n``).

    Returns
    -------
    torch.Tensor
        The transform, the shape of the input.

    Examples
    --------
    >>> spectrum = fft(image, axes=(-2, -1), unitary=True)
    """
    return dispatch(
        "fft",
        [input],
        None,
        _pos=[_flags(axes, input.ndim)],
        i=inverse,
        u=unitary,
        n=uncentred,
    )


@curated("fftmod")
def fftmod(input: torch.Tensor, axes: int | tuple[int, ...], *, inverse: bool = False):
    """Apply the alternating sign that centres a transform, along ``axes``.

    This is the multiplication ``fft`` would do for a centred transform, on
    its own, for data that has already been transformed.
    """
    return dispatch("fftmod", [input], None, _pos=[_flags(axes, input.ndim)], i=inverse)


@curated("fftshift")
def fftshift(input: torch.Tensor, axes: int | tuple[int, ...], *, inverse: bool = False):
    """Move the zero frequency to the middle along ``axes``, or back (``-i``)."""
    return dispatch("fftshift", [input], None, _pos=[_flags(axes, input.ndim)], i=inverse)


def ifft(
    input: torch.Tensor,
    axes: int | tuple[int, ...],
    *,
    unitary: bool = False,
    uncentred: bool = False,
) -> torch.Tensor:
    """The inverse Fourier transform along ``axes``.

    :func:`fft` with ``inverse=True``; BART has no separate command.
    """
    return fft(input, axes, inverse=True, unitary=unitary, uncentred=uncentred)


@curated("nufft")
def nufft(
    traj: torch.Tensor,
    input: torch.Tensor,
    *,
    adjoint: bool = False,
    inverse: bool = False,
    image_shape: tuple[int, ...] | None = None,
    image_dims: tuple[int, ...] | None = None,
    weights: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    toeplitz: bool | None = None,
    lowmem: bool = False,
    l2: float | None = None,
    maxiter: int | None = None,
    **extra,
) -> torch.Tensor:
    """The non-Cartesian transform along ``traj``, which FINUFFT computes.

    Parameters
    ----------
    traj : torch.Tensor
        Trajectory in grid units, of shape ``(..., samples, 3)``, as
        :func:`bartorch.tools.traj` produces.  It comes first because
        ``bart nufft`` takes it first.
    input : torch.Tensor
        An image for the forward transform, samples for the adjoint or the
        inverse.
    adjoint : bool
        Transform samples back to an image (``-a``).
    inverse : bool
        Solve for the image rather than transforming back (``-i``).
    image_shape : tuple of int, optional
        The image's shape for an adjoint or inverse transform (``-d``), when
        it cannot be read off the trajectory.
    weights : tensor, optional
        A diagonal in k-space (``-p``), applied on the way out and its
        conjugate on the way back.
    basis : tensor, optional
        A subspace basis over frames and coefficients (``-B``).
    toeplitz : bool, optional
        Apply the normal operator through the Toeplitz embedding.  ``False``
        passes ``--no-toeplitz``; ``None`` leaves BART its own default, which
        is on for an inverse and off otherwise.
    lowmem : bool
        Hold one set of frequencies of the point spread function at a time
        (``--lowmem``).
    l2 : float, optional
        Tikhonov weight for the inverse (``-l``).
    maxiter : int, optional
        Conjugate-gradient steps for the inverse (``-m``).
    **extra
        Further BART ``nufft`` flags, by name.

    Returns
    -------
    torch.Tensor
        Samples, or an image.

    Notes
    -----
    The transform is FINUFFT's: BART's own gridder is not reachable from this
    package.  :func:`bartorch.finufft.operators_built` says which answered.
    """
    flags: dict = dict(extra)
    if adjoint:
        flags["a"] = True
    if inverse:
        flags["i"] = True
    if image_dims is not None:
        # BART's own order, x:y:z, as the flag itself takes it.
        flags["d"] = tuple((list(image_dims) + [1, 1, 1])[:3])
    if image_shape is not None:
        # A C-order shape, reversed into BART's x:y:z.
        spatial = [int(n) for n in reversed(image_shape)]
        flags["d"] = tuple((spatial + [1, 1, 1])[:3])
    if weights is not None:
        flags["p"] = weights
    if basis is not None:
        flags["B"] = basis
    if toeplitz is False:
        flags["no_toeplitz"] = True
    if lowmem:
        flags["lowmem"] = True
    if l2 is not None:
        flags["l"] = l2
    if maxiter is not None:
        flags["m"] = maxiter
    return dispatch("nufft", [traj, input], None, **flags)
