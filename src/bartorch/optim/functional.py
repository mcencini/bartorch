"""The solvers as functions, for a reconstruction written in one expression.

Each function constructs the corresponding solver and calls it;
``optim.fista(y, A, term, maxiter=30)`` is ``optim.FISTA(term, maxiter=30)(y, A)``.
"""

from __future__ import annotations

import torch

from bartorch.optim.linear import ADMM, CG, FISTA, IST, NIHT, PRIDU
from bartorch.optim.pocs import POCS

__all__ = ["admm", "cg", "fista", "ist", "niht", "pocs", "pridu"]


def ist(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    r"""Solve a regularized least-squares problem by iterative soft thresholding.

    Minimizes :math:`\tfrac12 \| A x - y \|^2 + g(x)` by alternating a gradient
    step on the data-fidelity term with the proximal operator of ``g``.  Takes
    exactly one regularizer, whose linear transform must be the identity.

    Parameters
    ----------
    y : tensor
        Data of ``A.oshape``.
    A : LinearOperator
        The encoding operator.
    regularizers : Regularizer or ImplicitPrior, default=None
        The single term ``g``.
    x0 : tensor, default=None
        Warm start of ``A.ishape``; without one the iteration starts at zero.
    **settings
        Settings of :class:`~bartorch.optim.IST`, among them ``maxiter``
        (30), ``step`` and ``eigen``.

    Returns
    -------
    torch.Tensor
        Complex64 solution of ``A.ishape``.
    """
    return IST(regularizers, **settings)(y, A, x0)


def fista(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    r"""Solve a regularized least-squares problem by fast iterative soft thresholding.

    :func:`ist` with Nesterov momentum between the proximal step and the
    gradient step.  Takes exactly one regularizer, whose linear transform must
    be the identity.

    Parameters
    ----------
    y : tensor
        Data of ``A.oshape``.
    A : LinearOperator
        The encoding operator.
    regularizers : Regularizer or ImplicitPrior, default=None
        The single term ``g`` in
        :math:`\tfrac12 \| A x - y \|^2 + g(x)`.
    x0 : tensor, default=None
        Warm start of ``A.ishape``; without one the iteration starts at zero.
    **settings
        Settings of :class:`~bartorch.optim.FISTA`, among them ``maxiter``
        (30), ``step`` and ``eigen``.

    Returns
    -------
    torch.Tensor
        Complex64 solution of ``A.ishape``.
    """
    return FISTA(regularizers, **settings)(y, A, x0)


def admm(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    r"""Solve a regularized least-squares problem by alternating direction multipliers.

    Minimizes :math:`\tfrac12 \| A x - y \|^2 + \sum_j g_j(G_j x - b_j)` by
    splitting each term, so it takes any number of regularizers and, unlike
    :func:`ist` and :func:`fista`, terms whose linear transform :math:`G_j` is
    not the identity -- total variation among them -- and terms with auxiliary
    variables.  Each iteration solves its quadratic subproblem by conjugate
    gradients.

    Parameters
    ----------
    y : tensor
        Data of ``A.oshape``.
    A : LinearOperator
        The encoding operator.
    regularizers : Regularizer or ImplicitPrior, or an iterable of them, default=None
        The terms :math:`g_j`.
    x0 : tensor, default=None
        Warm start of ``A.ishape``; without one the iteration starts at zero.
    **settings
        Settings of :class:`~bartorch.optim.ADMM`, among them ``maxiter``
        (30), ``rho`` and ``cg_maxiter`` (10).

    Returns
    -------
    torch.Tensor
        Complex64 solution of ``A.ishape``.
    """
    return ADMM(regularizers, **settings)(y, A, x0)


def pridu(y: torch.Tensor, A, regularizers=None, *, x0=None, **settings):
    r"""Solve a regularized least-squares problem by a primal-dual iteration.

    Minimizes :math:`\tfrac12 \| A x - y \|^2 + \sum_j g_j(G_j x)` by
    Chambolle-Pock, keeping a dual variable per term whose linear transform is
    not the identity and applying the remaining term as a primal proximal step.
    Takes any number of regularizers, terms with a transform, and terms with
    auxiliary variables, and needs no inner solve.

    Parameters
    ----------
    y : tensor
        Data of ``A.oshape``.
    A : LinearOperator
        The encoding operator.
    regularizers : Regularizer or ImplicitPrior, or an iterable of them, default=None
        The terms :math:`g_j`.
    x0 : tensor, default=None
        Warm start of ``A.ishape``; without one the iteration starts at zero.
    **settings
        Settings of :class:`~bartorch.optim.PRIDU`, among them ``maxiter``
        (30), ``step`` and ``sigma_tau_ratio``.

    Returns
    -------
    torch.Tensor
        Complex64 solution of ``A.ishape``.
    """
    return PRIDU(regularizers, **settings)(y, A, x0)


def niht(y: torch.Tensor, A, regularizers, *, x0=None, **settings):
    r"""Solve a sparsity-constrained least-squares problem by normalized hard thresholding.

    Minimizes :math:`\tfrac12 \| A x - y \|^2` subject to a bound on the number
    of non-zero coefficients, taking
    :class:`~bartorch.priors.WaveletNIHT` and
    :class:`~bartorch.priors.ImageNIHT` terms and no others.

    Not usable: the underlying iteration applies the normal operator in place
    while the operator it is given asserts that its arguments are not aliased,
    so every solve terminates in an assertion.  See
    :meth:`bartorch.optim.NIHT.__call__`.

    Parameters
    ----------
    y : tensor
        Data of ``A.oshape``.
    A : LinearOperator
        The encoding operator.
    regularizers : WaveletNIHT or ImageNIHT, or an iterable of them
        The hard-thresholding terms.
    x0 : tensor, default=None
        Warm start of ``A.ishape``; without one the iteration starts at zero.
    **settings
        Settings of :class:`~bartorch.optim.NIHT`, among them ``maxiter``
        (30).

    Returns
    -------
    torch.Tensor
        Complex64 solution of ``A.ishape``.
    """
    return NIHT(regularizers, **settings)(y, A, x0)


def cg(y: torch.Tensor, A, lambda_: float = 0.0, *, x0=None, **settings):
    r"""Solve a linear least-squares problem by conjugate gradients.

    Minimizes :math:`\| A x - y \|^2 + \lambda \| x \|^2`, and with quadratic
    penalties :math:`\sum_i w_i \| G_i x - b_i \|^2` alongside it.  Takes
    :class:`~bartorch.optim.Tikhonov` penalties through ``terms=``; proximal
    regularizers are not accepted.

    Parameters
    ----------
    y : tensor
        Data of ``A.oshape``.
    A : LinearOperator
        The encoding operator.
    lambda_ : float, default=0.0
        Tikhonov weight on the image itself.
    x0 : tensor, default=None
        Warm start of ``A.ishape``; without one the iteration starts at zero.
    **settings
        Settings of :class:`~bartorch.optim.CG`, among them ``terms``,
        ``maxiter`` (30) and ``tol``.

    Returns
    -------
    torch.Tensor
        Complex64 solution of ``A.ishape``.

    Notes
    -----
    The solve is recorded for autograd when ``y`` requires a gradient; the
    backward pass is a second solve with the same normal operator.
    """
    return CG(lambda_, **settings)(y, A, x0)


def pocs(y: torch.Tensor, projections, *, x0=None, **settings):
    """Sweep a list of projections, ``optim.POCS(projections)(y, x0=x0)``.

    Unlike the solvers beside it this one has no encoding argument: what the
    sets are, and what the data is, belong to the projections themselves.

    Parameters
    ----------
    y : tensor
        What the iterate looks like.  Without ``x0`` the sweep starts at zero
        of its shape, where ``pocs_recon2`` starts.
    projections : sequence
        The sets to project onto, as :class:`~bartorch.optim.POCSBlock` takes
        them: a callable on a tensor, or a :mod:`bartorch.priors` term.
    x0 : tensor, default=None
        Where to start instead.
    **settings
        Settings of :class:`~bartorch.optim.POCS`, which is ``maxiter`` (50).

    Returns
    -------
    torch.Tensor
        The iterate the last sweep left.
    """
    return POCS(projections, **settings)(y, None, x0)
