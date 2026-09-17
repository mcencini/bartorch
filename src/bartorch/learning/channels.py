"""Complex tensors in the channel-first real layout networks and ``torchio`` use."""

from __future__ import annotations

import torch

__all__ = ["as_complex", "as_real"]


def as_real(input: torch.Tensor) -> torch.Tensor:
    """Real and imaginary parts of ``input`` stacked along a new leading axis.

    The pair is placed in front rather than in the trailing axis used by
    :func:`torch.view_as_real`, matching the channel-first convention of
    convolutional networks and of ``torchio.ScalarImage``.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor of any shape.  A real tensor is given a zero imaginary
        part, so that magnitude images and complex k-space can be carried in
        the same layout.

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``(2, *input.shape)``.

    Notes
    -----
    ``torchio.ScalarImage`` requires four axes, ``(channels, width, height,
    depth)``, so a two-dimensional image is unsqueezed to
    ``(2, height, width, 1)`` before it is passed on.

    Examples
    --------
    >>> volume = learning.as_real(image)                      # (2, x, y, z)
    >>> subject = torchio.Subject(image=torchio.ScalarImage(tensor=volume))
    """
    if input.is_complex():
        return torch.stack([input.real, input.imag])
    return torch.stack([input, torch.zeros_like(input)])


def as_complex(input: torch.Tensor) -> torch.Tensor:
    """Complex tensor formed from a leading axis of real and imaginary parts.

    Inverse of :func:`as_real`.

    Parameters
    ----------
    input : torch.Tensor
        Real tensor whose leading axis has length two.

    Returns
    -------
    torch.Tensor
        Complex tensor of shape ``input.shape[1:]``.
    """
    if input.is_complex():
        raise TypeError(f"as_complex takes the real pair, and {input.dtype} is complex already")
    if 0 == input.ndim or 2 != input.shape[0]:
        raise ValueError(
            f"the real and imaginary parts are the leading axis, so it has length two, "
            f"and {tuple(input.shape)} does not"
        )
    return torch.complex(input[0], input[1])
