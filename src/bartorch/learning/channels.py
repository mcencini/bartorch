"""A complex image as real channels, which is what a network and torchio take."""

from __future__ import annotations

import torch

__all__ = ["as_complex", "as_real"]


def as_real(input: torch.Tensor) -> torch.Tensor:
    """``input`` with its real and imaginary parts in a leading channel axis.

    A complex tensor of shape ``s`` becomes a real one of ``(2, *s)`` -- the
    pair in front rather than at the back, where :func:`torch.view_as_real`
    puts it, because that is where a convolution's channels are and where
    ``torchio``'s images carry theirs.  A real tensor is given a zero
    imaginary part, so that one k-space and one magnitude image can be
    carried the same way.

    Parameters
    ----------
    input : torch.Tensor
        Complex or real, of any shape.

    Returns
    -------
    torch.Tensor
        Real, of ``(2, *input.shape)``.  A ``torchio.ScalarImage`` wants four
        axes, so a slice is unsqueezed to ``(2, height, width, 1)`` before it
        is handed over.

    Examples
    --------
    >>> volume = learning.as_real(image)                      # (2, x, y, z)
    >>> subject = torchio.Subject(image=torchio.ScalarImage(tensor=volume))
    """
    if input.is_complex():
        return torch.stack([input.real, input.imag])
    return torch.stack([input, torch.zeros_like(input)])


def as_complex(input: torch.Tensor) -> torch.Tensor:
    """The complex tensor :func:`as_real` made ``input`` from.

    Parameters
    ----------
    input : torch.Tensor
        Real, with a leading axis of two.

    Returns
    -------
    torch.Tensor
        Complex, of ``input.shape[1:]``.
    """
    if input.is_complex():
        raise TypeError(f"as_complex takes the real pair, and {input.dtype} is complex already")
    if 0 == input.ndim or 2 != input.shape[0]:
        raise ValueError(
            f"the real and imaginary parts are the leading axis, so it is two long, "
            f"and {tuple(input.shape)} is not"
        )
    return torch.complex(input[0], input[1])
