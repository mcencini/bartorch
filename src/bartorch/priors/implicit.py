"""A denoiser in place of a regularization term."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ImplicitPrior"]


def _batched(apply, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """``apply`` item by item over a leading batch axis, when ``x`` has one."""
    if x.ndim == len(shape) + 1:
        return torch.stack([apply(item) for item in x])
    return apply(x)


class ImplicitPrior(nn.Module):
    """A denoiser substituted for a :mod:`bartorch.priors` regularizer.

    The proximal operator is ``denoiser(x, sigma)`` independently of the step,
    as plug-and-play regularization defines it, or ``denoiser(x)`` where no
    ``sigma`` is given.  Unlike a BART proximal operator this one is
    differentiable, so a solve containing it can be differentiated end to end.

    The denoiser receives a leading batch axis, of length one for a single
    image, and a complex image unchanged.  ``sigma`` is a
    :class:`torch.nn.Parameter`, frozen until ``requires_grad_()`` is called.

    ``transform`` is the :math:`G` of a term :math:`g(G x)`, which a
    regularizer built by BART also carries: the denoiser then acts on
    :math:`G`'s codomain rather than on the image, and the alternating-direction
    and primal-dual iterations split at :math:`G x` -- one auxiliary variable
    and one dual per term, and :math:`G^H G` in the x-update.  A prior learned
    in a domain the unknown is not in goes here: the contrast-weighted images a
    subspace basis makes of coefficient maps, say.  Without it :math:`G` is the
    identity and the denoiser sees the image.

    Parameters
    ----------
    denoiser : callable
        Called as ``denoiser(x)``, or as ``denoiser(x, sigma)`` when a
        ``sigma`` is given, on ``(batch, *shape)`` where ``shape`` is the
        image's, or ``transform``'s codomain when there is one.
        :class:`bartorch.learning.Denoiser` is what puts a network that takes
        real planes here.
    sigma : float, optional
        The noise level the denoiser is asked for, in the units its own
        convention states.
    transform : LinearOperator, optional
        :math:`G`, mapping the image to what the denoiser acts on.  Only the
        iterations that are given a term's transform use it; see
        :meth:`bartorch.priors.Regularizer.transform_is_identity`.

    Examples
    --------
    >>> optim.fista(y, A, priors.ImplicitPrior(to_complex_denoiser(DRUNet()), sigma=0.05))
    >>> priors.ImplicitPrior(denoiser, transform=linop.MultiplySum(basis, ...))
    """

    #: Takes a batch whole rather than item by item.
    _batches = True

    def __init__(self, denoiser, sigma: float | None = None, *, transform=None):
        super().__init__()
        if not callable(denoiser):
            raise TypeError(f"a denoiser is called as denoiser(x, sigma), and {denoiser!r} is not")
        self.denoiser = denoiser
        self.transform = transform
        self.sigma = (
            None if sigma is None else nn.Parameter(torch.tensor(float(sigma)), requires_grad=False)
        )

    def prox(self, x: torch.Tensor, gamma=1.0, *, image_shape=None) -> torch.Tensor:
        shape = None if image_shape is None else self.prox_shape(image_shape)
        single = shape is None or x.ndim == len(shape)
        batch = x[None] if single else x
        if self.sigma is None:
            out = self.denoiser(batch)
        else:
            sigma = self.sigma if self.sigma.requires_grad else float(self.sigma)
            out = self.denoiser(batch, sigma)
        return out[0] if single else out

    def prox_shape(self, image_shape) -> tuple[int, ...]:
        if self.transform is None:
            return tuple(image_shape)
        self._check(image_shape)
        return tuple(self.transform.oshape)

    def apply_transform(self, x: torch.Tensor, image_shape=None, mode: str = "forward"):
        if self.transform is None:
            return x
        from bartorch.linop.autograd import apply_adjoint, apply_forward, apply_normal

        if image_shape is None:
            image_shape = tuple(self.transform.ishape)
        self._check(image_shape)
        G = self.transform
        one, shape = {
            "forward": (apply_forward, tuple(G.ishape)),
            "adjoint": (apply_adjoint, tuple(G.oshape)),
            "normal": (apply_normal, tuple(G.ishape)),
        }[mode]
        return _batched(lambda v: one(G, v), x, shape)

    def transform_is_identity(self, image_shape) -> bool:
        return self.transform is None

    def rewind(self, image_shape) -> None:
        return None

    def _check(self, image_shape) -> None:
        """That the transform starts from the image the solve is over."""
        if tuple(self.transform.ishape) != tuple(image_shape):
            raise ValueError(
                f"{self!r}'s transform takes {tuple(self.transform.ishape)}, and the image is "
                f"{tuple(image_shape)}"
            )
