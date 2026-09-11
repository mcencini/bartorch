"""Rearranging an array, on axes rather than on bitmasks."""

from __future__ import annotations

import torch

from bartorch._operator import axes_flags
from bartorch.core.graph import dispatch
from bartorch.tools._call import curated

__all__ = ["flip"]


@curated("flip")
def flip(input: torch.Tensor, axes: int | tuple[int, ...]) -> torch.Tensor:
    """Reverse ``input`` along ``axes``.

    Parameters
    ----------
    input : torch.Tensor
        The array, C order.
    axes : int or tuple of int
        Which axes to reverse, as indices into ``input.shape``.

    Returns
    -------
    torch.Tensor
        The array, reversed.
    """
    return dispatch("flip", [input], None, _pos=[axes_flags(axes, input.ndim)])
