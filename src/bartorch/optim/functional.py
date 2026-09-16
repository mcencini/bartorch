"""The solvers as functions, for a reconstruction written in one expression.

``optim.fista(y, A, term, maxiter=30)`` is ``optim.FISTA(term, maxiter=30)(y, A)``.
"""

from __future__ import annotations

import torch

from bartorch.optim.linear import ADMM, CG, FISTA, IST, NIHT, PRIDU

__all__ = ["admm", "cg", "fista", "ist", "niht", "pridu"]


def ist(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Iterative soft thresholding.  See :class:`~bartorch.optim.IST`."""
    return IST(regularizers, **settings)(y, A, x0)


def fista(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Fast iterative soft thresholding.  See :class:`~bartorch.optim.FISTA`."""
    return FISTA(regularizers, **settings)(y, A, x0)


def admm(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Alternating direction method of multipliers.  See :class:`~bartorch.optim.ADMM`."""
    return ADMM(regularizers, **settings)(y, A, x0)


def pridu(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Primal-dual iteration.  See :class:`~bartorch.optim.PRIDU`."""
    return PRIDU(regularizers, **settings)(y, A, x0)


def niht(y: torch.Tensor, A, regularizers, *, x0=None, **settings):
    """Normalized iterative hard thresholding.  See :class:`~bartorch.optim.NIHT`.

    Takes :class:`~bartorch.priors.WaveletNIHT` and
    :class:`~bartorch.priors.ImageNIHT` terms and nothing else.
    """
    return NIHT(regularizers, **settings)(y, A, x0)


def cg(y: torch.Tensor, A, lambda_: float = 0.0, *, x0=None, **settings):
    """Conjugate gradients.  See :class:`~bartorch.optim.CG`.

    Takes the quadratic penalties described by
    :class:`~bartorch.optim.Tikhonov`; proximal terms are not accepted.
    """
    return CG(lambda_, **settings)(y, A, x0)
