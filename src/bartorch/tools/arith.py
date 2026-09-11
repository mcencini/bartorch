"""Arithmetic over an array, on axes rather than on bitmasks."""

from __future__ import annotations

import torch

from bartorch._operator import axes_flags
from bartorch.core.graph import dispatch
from bartorch.tools._call import curated

__all__ = ["avg", "rss", "scale", "std", "var"]


def _over(command: str, input: torch.Tensor, axes, **extra) -> torch.Tensor:
    return dispatch(command, [input], None, _pos=[axes_flags(axes, input.ndim)], **extra)


@curated("rss")
def rss(input: torch.Tensor, axes: int | tuple[int, ...]) -> torch.Tensor:
    """The root of the sum of squares over ``axes``.

    The usual way to combine coil images without sensitivities.
    """
    return _over("rss", input, axes)


@curated("avg")
def avg(input: torch.Tensor, axes: int | tuple[int, ...], *, weighted: bool = False):
    """The mean over ``axes``, or the weighted mean (``-w``)."""
    return _over("avg", input, axes, w=weighted)


@curated("std")
def std(input: torch.Tensor, axes: int | tuple[int, ...]) -> torch.Tensor:
    """The standard deviation over ``axes``."""
    return _over("std", input, axes)


@curated("var")
def var(input: torch.Tensor, axes: int | tuple[int, ...]) -> torch.Tensor:
    """The variance over ``axes``."""
    return _over("var", input, axes)


@curated("scale")
def scale(input: torch.Tensor, factor: complex) -> torch.Tensor:
    """``input`` multiplied by ``factor``.

    BART reads the factor from the command line, so it is a number and not an
    array, which is what a wrapper reading the argument table as files got
    wrong.
    """
    return dispatch("scale", [input], None, _pos=[factor])
