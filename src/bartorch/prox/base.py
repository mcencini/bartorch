"""The proximal operator every concrete one is."""

from __future__ import annotations

import abc

import torch

__all__ = ["Prox"]


class Prox(abc.ABC):
    """The proximal operator of a function, as a map on tensors.

    ``prox(alpha, x)`` is ``argmin_z  g(z) + |z - x|^2 / (2 alpha)``, which is
    the step a proximal algorithm takes for the part of the objective it does
    not differentiate.

    Every one here is written in torch rather than taken from BART, and the
    reason is what they are for: a proximal step inside a Python loop is one
    that can be unrolled and differentiated through.  BART's own regularizers
    are reachable, and faster, through :func:`bartorch.tools.pics` -- they just
    cannot be part of a torch graph.

    Attributes
    ----------
    shape : tuple of int
        What it operates on.
    """

    shape: tuple[int, ...]

    @abc.abstractmethod
    def __call__(self, alpha: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The proximal step of size ``alpha`` at ``x``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.shape})"
