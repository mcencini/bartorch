"""Nonlinear inverse problems by BART's iteratively regularized Gauss-Newton.

BART has the method in two forms, and :class:`IRGNM` is both.  ``irgnm``
solves each linearized problem with its own conjugate gradients and nothing
else, which is what ``nlinv`` runs and what ``IRGNM`` does without an inner
solver.  ``irgnm2`` pays an extra application of the derivative and hands the
problem to a generic regularized least-squares solver, which is how a
regularized ``nlinv`` or ``moba`` works; ``inner=`` is that form, with the
outer loop written out here so any solver in :mod:`bartorch.optim` can take
it.
"""

from __future__ import annotations

import copy

import torch

from bartorch._dispatch import BartError, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand

__all__ = ["IRGNM"]

#: What a string names, for a caller who does not want to build a solver.
_BY_NAME = ("cg", "ist", "fista", "admm", "pridu")


def _inner_solver(inner):
    """The solver an ``inner=`` argument stands for."""
    from bartorch.optim import linear

    if isinstance(inner, str):
        if inner not in _BY_NAME:
            raise ValueError(f"no inner solver called {inner!r}; one of {list(_BY_NAME)}")
        try:
            return {
                "cg": linear.CG,
                "ist": linear.IST,
                "fista": linear.FISTA,
                "admm": linear.ADMM,
                "pridu": linear.PRIDU,
            }[inner]()
        except ValueError as exc:
            raise ValueError(
                f"{inner!r} is a regularized solver and has nothing to regularize with; "
                f"build it with its term, as inner=optim.{inner.upper()}(prox.Wavelet(...))"
            ) from exc
    if not isinstance(inner, linear._Solver):
        raise TypeError(
            f"inner takes one of {list(_BY_NAME)} or a solver from bartorch.optim, "
            f"not {type(inner).__name__}"
        )
    return inner


def _at(solver, alpha: float):
    """``solver`` as this Newton step needs it.

    Two things change per step.  The Tikhonov weight goes on the normal
    operator, which is where ``lsqr2_create(..., lambda = alpha, ...)`` puts
    it and where ``cclambda`` puts it.

    And a step-size solver power-iterates.  The normal operator of a Newton
    step is ``DF^H DF + alpha I``, whose largest eigenvalue changes with the
    linearization point *and* with alpha, so a fixed step diverges the moment
    it exceeds ``2 / L``.  BART's own inner FISTA does not offer the choice --
    ``moba/iter_l1.c``'s ``inverse_fista`` computes ``alpha + power(20,
    normal)`` and scales by it every single step -- so neither does this.
    """
    step = copy.copy(solver)
    step.cclambda = float(alpha)
    if hasattr(step, "eigen"):
        step.eigen = True
    return step


class IRGNM:
    """Iteratively regularized Gauss-Newton for ``F(x) = y``.

    Each step linearizes at the current point and solves

    ``min_u ||DF u - r||^2 + alpha ||u||^2 + R(u)``

    where ``r`` is the residual carried to the linearization; the weight is
    then divided by ``redu``, down to ``alpha_min``.  ``R`` is whatever the
    inner solver regularizes with, and is nothing at all for conjugate
    gradients.

    Parameters
    ----------
    iterations : int
        Gauss-Newton steps.
    alpha : float
        Initial Tikhonov weight.
    alpha_min : float
        What the weight decays towards.
    alpha_min0 : float
        A floor the decayed weight is never taken below.  Only the second
        form has it; BART's ``irgnm`` does not.
    redu : float
        Factor the weight is divided by after each step.
    cg_maxiter : int
        Conjugate-gradient iterations per step, for the built-in solver.
    cg_tol : float
        Conjugate-gradient tolerance per step, for the built-in solver.
    inner : str, solver or None
        The solver for the linearized problem.  ``None`` runs BART's first
        form, entirely inside the library, which is what ``nlinv`` runs.
        Anything else runs the second form: ``"cg"``, ``"ist"``, ``"fista"``,
        ``"admm"``, ``"pridu"``, or a configured solver from
        :mod:`bartorch.optim`, whose regularizers become the ``R`` above.

    Examples
    --------
    Plain, which is ``nlinv``:

    >>> IRGNM(iterations=8)(kspace, F, x0=start)

    Wavelet-regularized, which is what ``moba -l1`` runs:

    >>> IRGNM(inner=optim.FISTA(prox.Wavelet((-1, -2), 0.001), maxiter=30))(
    ...     kspace, F, x0=start
    ... )

    Several terms at once, which is ADMM's job:

    >>> IRGNM(
    ...     inner=optim.ADMM(
    ...         [prox.Wavelet((-1, -2), 0.001), prox.TotalVariation((-1, -2), 0.01)]
    ...     )
    ... )(kspace, F, x0=start)

    Notes
    -----
    The two forms are the same method and not the same arithmetic.  The first
    carries ``alpha (xref - x)`` into the right-hand side and solves for a
    step; the second shifts by ``xref``, carries an extra ``DF (x - xref)``
    into the residual, and solves for the iterate itself.  They agree in exact
    arithmetic and differ in the last bits, so a run with ``inner=`` will not
    reproduce one without it -- but ``inner="cg"`` reproduces
    ``iter4_irgnm2`` exactly, which is what the suite holds it to.
    """

    def __init__(
        self,
        *,
        iterations: int = 8,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        alpha_min0: float = 0.0,
        redu: float = 2.0,
        cg_maxiter: int = 30,
        cg_tol: float = 0.0,
        inner=None,
    ):
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_min0 = float(alpha_min0)
        self.redu = float(redu)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_tol = float(cg_tol)
        self.inner = None if inner is None else _inner_solver(inner)

    def __call__(
        self, y: torch.Tensor, F, x0: torch.Tensor, xref: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fit ``F(x) = y`` starting from ``x0``.

        Parameters
        ----------
        y : torch.Tensor
            Data of ``F.oshape``.
        F : NonlinearOperator or LinearOperator
            The forward model; a linear one is converted with
            :meth:`~bartorch.linop.LinearOperator.to_nonlinear`, and a model
            of several unknowns with
            :meth:`~bartorch.nlop.NonlinearOperator.flatten`.
        x0 : torch.Tensor
            Starting point of ``F.ishape``.
        xref : torch.Tensor, optional
            Regularization centre of ``F.ishape``.  Without one the steps are
            regularized towards zero, which is what BART does.

        Returns
        -------
        torch.Tensor
            Complex64 solution of ``F.ishape``.
        """
        from bartorch.linop.base import LinearOperator

        if isinstance(F, LinearOperator):
            F = F.to_nonlinear()
        op = F._bart()
        y = as_operand(y, op.oshape, "y")
        x = as_operand(x0, op.ishape, "x0").clone()
        ref = as_operand(xref, op.ishape, "xref") if xref is not None else None
        if self.inner is not None:
            return self._in_python(y, op, x, ref)
        return self._in_library(y, op, x, ref)

    # --- BART's first form, entirely inside the library --------------------

    def _in_library(self, y, op, x, ref) -> torch.Tensor:
        with _lock, _on_device(op.device or y.device):
            code = library().bartorch_irgnm(
                op._h.ptr,
                self.iterations,
                self.alpha,
                self.alpha_min,
                self.redu,
                self.cg_maxiter,
                self.cg_tol,
                x.data_ptr(),
                y.data_ptr(),
                ref.data_ptr() if ref is not None else None,
            )
        if code != 0:
            raise BartError("Gauss-Newton solve failed; see the log for BART's message")
        return x

    def in_library(
        self, y: torch.Tensor, F, x0: torch.Tensor, xref: torch.Tensor | None = None
    ) -> torch.Tensor:
        """BART's second form with its own conjugate gradients, ``iter4_irgnm2``.

        What :meth:`__call__` with ``inner="cg"`` is held against: the loop
        below is written out in Python so the inner problem can go elsewhere,
        and this is the same loop inside the library.
        """
        from bartorch.linop.base import LinearOperator

        if isinstance(F, LinearOperator):
            F = F.to_nonlinear()
        op = F._bart()
        y = as_operand(y, op.oshape, "y")
        x = as_operand(x0, op.ishape, "x0").clone()
        ref = as_operand(xref, op.ishape, "xref") if xref is not None else None
        with _lock, _on_device(op.device or y.device):
            code = library().bartorch_irgnm2(
                op._h.ptr,
                self.iterations,
                self.alpha,
                self.alpha_min,
                self.alpha_min0,
                self.redu,
                self.cg_maxiter,
                self.cg_tol,
                x.data_ptr(),
                y.data_ptr(),
                ref.data_ptr() if ref is not None else None,
            )
        if code != 0:
            raise BartError("Gauss-Newton solve failed; see the log for BART's message")
        return x

    # --- BART's second form, with the inner problem anywhere ---------------

    def _in_python(self, y, op, x, ref) -> torch.Tensor:
        """``irgnm2``, written out.

        Every line below is one of ``italgos.c``'s, in its order.  The
        arithmetic is a scale and an add, which ``vecops.c`` does with the
        same kernel torch does, so the bits are the library's -- that is what
        ``inner="cg"`` against :meth:`in_library` checks.
        """
        alpha = self.alpha
        # The derivative is a view of the operator's own, so it is built once
        # and follows the point every forward call moves it to.
        jacobian = op.jacobian()
        for _ in range(self.iterations):
            # r = y - F(x), with the derivative fixed at this x.
            r = y - op.forward(x)

            if ref is not None:
                x = x - ref

            # The linearization is carried to the reference: what is solved
            # for is the iterate, not the step.
            r = r + op.derivative(x)

            x = _at(self.inner, alpha)(r, jacobian)

            if ref is not None:
                x = x + ref

            alpha = (alpha - self.alpha_min) / self.redu + self.alpha_min
            if alpha < self.alpha_min0:
                alpha = self.alpha_min0
        return x

    def __repr__(self) -> str:
        inner = "" if self.inner is None else f", inner={type(self.inner).__name__}(...)"
        return (
            f"IRGNM(iterations={self.iterations}, alpha={self.alpha}, "
            f"alpha_min={self.alpha_min}, redu={self.redu}{inner})"
        )
