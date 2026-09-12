"""The solvers as functions, for a reconstruction written in one line.

``optim.FISTA(term, maxiter=30)(y, A)`` is a solver configured and then
called, which is the shape to keep when the same configuration is used twice
-- inside a training loop, say.  For the other case, where a reconstruction is
run once, these are the same thing said in one expression::

    x = optim.fista(y, A, term, maxiter=30)

They take one more kind of argument than the classes' documentation suggests,
which is the point of having them: a ``deepinv`` prior or denoiser goes
wherever a :mod:`bartorch.prox` term goes.  A plug-and-play reconstruction is
then the same call with the denoiser in place of the term::

    from deepinv.models import DRUNet, to_complex_denoiser
    x = optim.admm(y, A, to_complex_denoiser(DRUNet()), g_param=0.03)

What runs is the iteration in :mod:`bartorch.optim.iterators`, which is BART's
step for step -- so a plug-and-play solve is BART's ADMM with the threshold
replaced, and not a second implementation of it.
"""

from __future__ import annotations

import torch

from bartorch.optim.linear import ADMM, CG, FISTA, IST, PRIDU
from bartorch.prox.base import Regularizer

__all__ = ["admm", "cg", "fista", "ist", "pridu"]


def _priors(regularizers, g_param: float | None):
    """``regularizers`` with anything that is not a term wrapped so it answers
    like one."""
    if regularizers is None:
        return None

    one = isinstance(regularizers, Regularizer) or not isinstance(regularizers, (list, tuple))
    terms = [regularizers] if one else list(regularizers)

    wrapped = []
    for term in terms:
        if isinstance(term, Regularizer):
            # A term carries its own weight; `g_param` is for the others, and
            # it passes this one by.
            wrapped.append(term)
        else:
            from bartorch.optim.iterators import AsTerm

            wrapped.append(term if isinstance(term, AsTerm) else AsTerm(term, g_param))

    if g_param is not None and all(isinstance(t, Regularizer) for t in wrapped):
        raise ValueError(
            "g_param is a deepinv prior's own parameter -- a denoiser's noise level, say; "
            f"{wrapped[0]!r} carries its weight"
        )
    return wrapped[0] if one else wrapped


def ist(y: torch.Tensor, A, regularizers=None, *, x0=None, g_param=None, **settings):
    """Iterative soft thresholding.  See :class:`~bartorch.optim.IST`."""
    return IST(_priors(regularizers, g_param), **settings)(y, A, x0)


def fista(y: torch.Tensor, A, regularizers=None, *, x0=None, g_param=None, **settings):
    """Fast iterative soft thresholding.  See :class:`~bartorch.optim.FISTA`."""
    return FISTA(_priors(regularizers, g_param), **settings)(y, A, x0)


def admm(y: torch.Tensor, A, regularizers=None, *, x0=None, g_param=None, **settings):
    """Alternating directions.  See :class:`~bartorch.optim.ADMM`.

    The one that takes several terms, so a plug-and-play prior and a term of
    BART's own can be split apart in the same solve.
    """
    return ADMM(_priors(regularizers, g_param), **settings)(y, A, x0)


def pridu(y: torch.Tensor, A, regularizers=None, *, x0=None, g_param=None, **settings):
    """Primal and dual.  See :class:`~bartorch.optim.PRIDU`."""
    return PRIDU(_priors(regularizers, g_param), **settings)(y, A, x0)


def cg(y: torch.Tensor, A, lambda_: float = 0.0, *, x0=None, **settings):
    """Conjugate gradients.  See :class:`~bartorch.optim.CG`.

    No prior goes here: this is the one solver with no proximal operator in
    it, and its penalties are the quadratic ones
    :class:`~bartorch.optim.Tikhonov` describes.
    """
    return CG(lambda_, **settings)(y, A, x0)
