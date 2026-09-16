"""A denoiser in place of a regularization term."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ImplicitPrior"]


class ImplicitPrior(nn.Module):
    """A denoiser substituted for a :mod:`bartorch.priors` regularizer.

    The proximal operator is ``denoiser(x, sigma)`` independently of the step,
    as plug-and-play regularization defines it, or ``denoiser(x)`` where no
    ``sigma`` is given.  Unlike a BART proximal operator this one is
    differentiable, so a solve containing it can be differentiated end to end.

    The denoiser receives a leading batch axis, of length one for a single
    image, and a complex image unchanged.  ``sigma`` is a
    :class:`torch.nn.Parameter`, frozen until ``requires_grad_()`` is called.

    Examples
    --------
    >>> optim.fista(y, A, priors.ImplicitPrior(to_complex_denoiser(DRUNet()), sigma=0.05))
    """

    #: Takes a batch whole rather than item by item.
    _batches = True

    def __init__(self, denoiser, sigma: float | None = None):
        super().__init__()
        if not callable(denoiser):
            raise TypeError(f"a denoiser is called as denoiser(x, sigma), and {denoiser!r} is not")
        self.denoiser = denoiser
        self.sigma = (
            None if sigma is None else nn.Parameter(torch.tensor(float(sigma)), requires_grad=False)
        )

    def prox(self, x: torch.Tensor, gamma=1.0, *, image_shape=None) -> torch.Tensor:
        single = image_shape is None or x.ndim == len(image_shape)
        batch = x[None] if single else x
        if self.sigma is None:
            out = self.denoiser(batch)
        else:
            sigma = self.sigma if self.sigma.requires_grad else float(self.sigma)
            out = self.denoiser(batch, sigma)
        return out[0] if single else out

    def prox_shape(self, image_shape) -> tuple[int, ...]:
        return tuple(image_shape)

    def apply_transform(self, x: torch.Tensor, image_shape=None, mode: str = "forward"):
        return x

    def transform_is_identity(self, image_shape) -> bool:
        return True

    def rewind(self, image_shape) -> None:
        return None
