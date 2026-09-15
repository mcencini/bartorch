"""The solvers as functions, for a reconstruction written in one expression.

``optim.fista(y, A, term, maxiter=30)`` is ``optim.FISTA(term,
maxiter=30)(y, A)``.  They take one argument the classes do not: a ``deepinv``
prior or a bare denoiser goes wherever a :mod:`bartorch.priors` term goes, with
``g_param`` as its own parameter.  What runs is the class.
"""

from __future__ import annotations

import torch

from bartorch.optim.linear import ADMM, CG, FISTA, IST, NIHT, PRIDU, EulerMaruyama
from bartorch.priors.base import Regularizer

__all__ = ["admm", "cg", "eulermaruyama", "fista", "irgnm", "ist", "niht", "pridu"]


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
            from bartorch.optim._iterators import AsTerm

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


def niht(y: torch.Tensor, A, regularizers, *, x0=None, **settings):
    """Normalized iterative hard thresholding.  See :class:`~bartorch.optim.NIHT`.

    Takes :class:`~bartorch.priors.WaveletNIHT` and
    :class:`~bartorch.priors.ImageNIHT` terms and nothing else.
    """
    return NIHT(regularizers, **settings)(y, A, x0)


def eulermaruyama(y: torch.Tensor, A, regularizers=None, *, x0=None, g_param=None, **settings):
    """Euler-Maruyama sampling.  See :class:`~bartorch.optim.EulerMaruyama`.

    ``step`` is required: ``pics`` supplies no default for this iteration.
    """
    return EulerMaruyama(_priors(regularizers, g_param), **settings)(y, A, x0)


def cg(y: torch.Tensor, A, lambda_: float = 0.0, *, x0=None, **settings):
    """Conjugate gradients.  See :class:`~bartorch.optim.CG`.

    No prior goes here: this is the one solver with no proximal operator in
    it, and its penalties are the quadratic ones
    :class:`~bartorch.optim.Tikhonov` describes.
    """
    return CG(lambda_, **settings)(y, A, x0)


def irgnm(y: torch.Tensor, F, *, x0=None, xref=None, inner=None, **settings):
    """Gauss-Newton for a nonlinear ``F``.  See :class:`~bartorch.optim.IRGNM`.

    ``F`` is a :class:`~bartorch.nlop.NonlinearOperator` rather than a linear
    encoding, and ``inner`` a configured solver from :mod:`bartorch.optim`.
    """
    from bartorch.optim.nonlinear import IRGNM

    return IRGNM(inner=inner, **settings)(y, F, x0=x0, xref=xref)
