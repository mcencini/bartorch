"""A denoiser where a regularization term goes."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ImplicitPrior"]


class ImplicitPrior(nn.Module):
    """A denoiser standing where a :mod:`bartorch.priors` term goes.

    Its proximal operator is ``denoiser(x, sigma)`` whatever the step, as
    plug-and-play takes it, or ``denoiser(x)`` without a ``sigma``.  The
    denoiser sees a leading batch axis, of one for a single image, and a
    complex image as it is.  ``sigma`` is a parameter, frozen until
    ``requires_grad_()``.

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
