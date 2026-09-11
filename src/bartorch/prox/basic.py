"""Proximal operators of the functions a reconstruction usually penalises."""

from __future__ import annotations

import torch

from bartorch.prox.base import Prox

__all__ = ["L1", "L2Ball", "L2Squared", "Stack", "Zero"]


def soft_threshold(x: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    """Shrink each entry's magnitude by ``threshold``, keeping its phase.

    The proximal operator of the l1 norm.  For a complex tensor the shrinkage
    is on the magnitude and the phase is untouched, which is what makes it the
    right thing for an image rather than for its real and imaginary parts
    separately.
    """
    magnitude = x.abs()
    scale = torch.clamp(magnitude - threshold, min=0)
    # Where the magnitude is zero the direction is undefined and the answer is
    # zero either way; the clamp keeps the division finite.
    return x * (scale / torch.clamp(magnitude, min=torch.finfo(magnitude.dtype).tiny))


class Zero(Prox):
    """The proximal operator of the zero function, which is the identity.

    What an unregularised solve uses, so that one loop serves both.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)

    def __call__(self, alpha, x):
        return x


class L1(Prox):
    """The proximal operator of ``weight * ||x||_1``: soft thresholding.

    Parameters
    ----------
    shape : tuple of int
        What it operates on.
    weight : float
        The penalty's weight.

    Examples
    --------
    >>> g = L1(image.shape, weight=0.01)
    >>> g(0.5, image)   # a step of half the size
    """

    def __init__(self, shape, weight: float = 1.0):
        self.shape = tuple(shape)
        self.weight = float(weight)

    def __call__(self, alpha, x):
        return soft_threshold(x, self.weight * alpha)


class L2Squared(Prox):
    """The proximal operator of ``weight * ||x - centre||^2``, which shrinks toward the centre.

    Tikhonov regularisation as a proximal step.  Without a centre it shrinks
    toward zero.
    """

    def __init__(self, shape, weight: float = 1.0, centre: torch.Tensor | None = None):
        self.shape = tuple(shape)
        self.weight = float(weight)
        self.centre = centre

    def __call__(self, alpha, x):
        scale = 1.0 / (1.0 + 2.0 * self.weight * alpha)
        if self.centre is None:
            return x * scale
        return self.centre + (x - self.centre) * scale


class L2Ball(Prox):
    """The proximal operator of the indicator of ``||x - centre|| <= radius``.

    A projection, so the step size does not enter.  This is the constraint a
    basis-pursuit formulation puts on the residual.
    """

    def __init__(self, shape, radius: float = 1.0, centre: torch.Tensor | None = None):
        self.shape = tuple(shape)
        self.radius = float(radius)
        self.centre = centre

    def __call__(self, alpha, x):
        offset = x if self.centre is None else x - self.centre
        norm = offset.flatten().norm()
        scale = torch.clamp(self.radius / torch.clamp(norm, min=1e-30), max=1.0)
        projected = offset * scale
        return projected if self.centre is None else self.centre + projected


class Stack(Prox):
    """Several proximal operators, each on its own part of a flattened input.

    What a problem with more than one penalty needs, when the penalties do not
    share a variable.
    """

    def __init__(self, proxes):
        self.proxes = tuple(proxes)
        if not self.proxes:
            raise ValueError("a stack needs at least one proximal operator")
        self._sizes = [int(torch.tensor(p.shape).prod()) for p in self.proxes]
        self.shape = (sum(self._sizes),)

    def __call__(self, alpha, x):
        flat = x.reshape(-1)
        pieces, at = [], 0
        for prox, size in zip(self.proxes, self._sizes, strict=True):
            piece = flat[at : at + size].reshape(prox.shape)
            pieces.append(prox(alpha, piece).reshape(-1))
            at += size
        return torch.cat(pieces).reshape(x.shape)
