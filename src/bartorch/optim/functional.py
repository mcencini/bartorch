"""The solvers as functions, for a reconstruction written in one expression.

``optim.fista(y, A, term, maxiter=30)`` is ``optim.FISTA(term, maxiter=30)(y, A)``.
"""

from __future__ import annotations

import torch

from bartorch.optim.linear import ADMM, CG, FISTA, IST, NIHT, PRIDU

__all__ = ["admm", "cg", "fista", "irgnm", "ist", "niht", "pridu"]


def ist(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Iterative soft thresholding.  See :class:`~bartorch.optim.IST`."""
    return IST(regularizers, **settings)(y, A, x0)


def fista(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Fast iterative soft thresholding.  See :class:`~bartorch.optim.FISTA`."""
    return FISTA(regularizers, **settings)(y, A, x0)


def admm(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Alternating directions.  See :class:`~bartorch.optim.ADMM`."""
    return ADMM(regularizers, **settings)(y, A, x0)


def pridu(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    """Primal and dual.  See :class:`~bartorch.optim.PRIDU`."""
    return PRIDU(regularizers, **settings)(y, A, x0)


def niht(y: torch.Tensor, A, regularizers, *, x0=None, **settings):
    """Normalized iterative hard thresholding.  See :class:`~bartorch.optim.NIHT`.

    Takes :class:`~bartorch.priors.WaveletNIHT` and
    :class:`~bartorch.priors.ImageNIHT` terms and nothing else.
    """
    return NIHT(regularizers, **settings)(y, A, x0)


def cg(y: torch.Tensor, A, lambda_: float = 0.0, *, x0=None, **settings):
    """Conjugate gradients.  See :class:`~bartorch.optim.CG`.

    Its penalties are the quadratic ones :class:`~bartorch.optim.Tikhonov`
    describes; no proximal term goes here.
    """
    return CG(lambda_, **settings)(y, A, x0)


def irgnm(y: torch.Tensor, F, *, x0=None, xref=None, inner=None, **settings):
    """Gauss-Newton for a nonlinear ``F``.  See :class:`~bartorch.optim.IRGNM`.

    ``F`` is a :class:`~bartorch.nlop.NonlinearOperator` rather than a linear
    encoding, and ``inner`` a configured solver from :mod:`bartorch.optim`.
    """
    from bartorch.optim.nonlinear import IRGNM

    return IRGNM(inner=inner, **settings)(y, F, x0=x0, xref=xref)
