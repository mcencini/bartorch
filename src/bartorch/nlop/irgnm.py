"""Nonlinear inverse problems by BART's iteratively regularized Gauss-Newton.

BART has the method in two forms, and :class:`IRGNM` is both.  ``irgnm``
solves each linearized problem with its own conjugate gradients, which is what
``nlinv`` runs and what ``IRGNM`` does without an inner solver.  ``irgnm2``
hands the problem to a generic regularized least-squares solver, which is how a
regularized ``nlinv`` or ``moba`` works; ``inner=`` is that form, with the outer
loop written out here.  :meth:`IRGNM.operator` is BART's own step of ``nlinv``
as an operator a network is built of.
"""

from __future__ import annotations

import copy

import torch

from bartorch._dispatch import BartError, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand

__all__ = ["IRGNM", "irgnm"]


def _inner_solver(inner):
    """``inner`` checked to be a configured solver, which is all it may be."""
    from bartorch.optim import linear

    if isinstance(inner, str):
        raise TypeError(
            f"inner takes a configured solver, not the name of one: pass "
            f"optim.{inner.upper()}() rather than {inner!r}, so that its terms and "
            "settings are visible where the solve is written"
        )
    if not isinstance(inner, linear._Solver):
        raise TypeError(f"inner takes a solver from bartorch.optim, not {type(inner).__name__}")
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
    inner : solver or None
        A configured solver from :mod:`bartorch.optim` for the linearized
        problem, whose regularizers become the ``R`` above.  ``None`` runs
        BART's first form, entirely inside the library, which is what
        ``nlinv`` runs.

    Examples
    --------
    Plain, which is ``nlinv``:

    >>> nlop.IRGNM(iterations=8)(kspace, F, x0=start)

    The same method with the linearized problem written out here, which is
    ``iter4_irgnm2`` to the bit:

    >>> nlop.IRGNM(iterations=8, inner=optim.CG())(kspace, F, x0=start)

    Wavelet-regularized, which is what ``moba -l1`` runs:

    >>> nlop.IRGNM(inner=optim.FISTA(priors.Wavelet((-1, -2), 0.001), maxiter=30))(
    ...     kspace, F, x0=start
    ... )

    Several terms at once, which is ADMM's job:

    >>> nlop.IRGNM(
    ...     inner=optim.ADMM(
    ...         [priors.Wavelet((-1, -2), 0.001), priors.TotalVariation((-1, -2), 0.01)]
    ...     )
    ... )(kspace, F, x0=start)

    Notes
    -----
    The two forms are the same method and not the same arithmetic.  The first
    carries ``alpha (xref - x)`` into the right-hand side and solves for a
    step; the second shifts by ``xref``, carries an extra ``DF (x - xref)``
    into the residual, and solves for the iterate itself.  They agree in exact
    arithmetic and differ in the last bits, so a run with ``inner=`` will not
    reproduce one without it -- but ``inner=optim.CG()`` reproduces
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
        return self._first_form(y, op, x, ref)

    # --- BART's first form, entirely inside the library --------------------

    def _first_form(self, y, op, x, ref) -> torch.Tensor:
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

    def _in_library(
        self, y: torch.Tensor, F, x0: torch.Tensor, xref: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``iter4_irgnm2`` with BART's own conjugate gradients: the reference
        ``inner=optim.CG()`` is held against."""
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

    # --- BART's whole step, as an operator ---------------------------------

    def operator(self, F, *, batch: int = 1, cg_lambda: float = 0.0):
        """This schedule as one operator ``(y, xn, x0, alpha) -> x``, differentiable by all four.

        ``iterations`` steps, the weight decaying by ``redu`` towards
        ``alpha_min``, each a conjugate-gradient solve whose backward pass is
        another.  Steps chain into one operator with
        :func:`~bartorch.nlop.chain`.

        ``F`` a :class:`~bartorch.nlop.NonlinearSense` is BART's own step of
        ``nlinv`` (``noir/model_net.c``); any other model is that same
        expression assembled over the derivative the model supplies as a
        function of the point (:attr:`~bartorch.nlop.NonlinearOperator.bundle`),
        and a model that supplies none solves through :meth:`__call__` instead.

        ``xn``, ``x0`` and the answer are the model's unknowns in one flat
        vector, which ``split()`` and ``join()`` read and write, and ``alpha``
        may be a number.  ``cg_lambda`` is the inner solve's ``l2lambda``.

        Notes
        -----
        For ``NonlinearSense``, ``y`` is coil images, which ``prepare()`` makes
        from k-space and a pattern; applying it is what gives the model its
        pattern, and a step refuses until then.  ``batch`` copies of the model
        sit on BART's batch axis, the leading axis of every argument, and only
        BART's own model carries one.

        BART's default coil weighting, ``b = 32``, puts part of the coil half's
        gradient below float32's smallest normal number, where ``checkeps``
        leaves the solve untouched and the gradient is zeros; a gradient that
        has to mean something there wants ``sobolev=(220.0, 8.0)``.  The network
        model fits the coils on the image's grid, so a ``NonlinearSense`` must
        have ``oversampling_coils=1.0`` and none of ``optimized``,
        ``oversampled_coils`` or a separate coefficient shape.

        Examples
        --------
        >>> F = nlop.CartesianSense((coils, 256, 256), sobolev=(220.0, 8.0))
        >>> cell = nlop.IRGNM(iterations=1).operator(F, batch=4)
        >>> y = cell.prepare()(kspace, pattern)
        >>> x1 = cell(y, cell.start(batch=4), cell.start(batch=4), 1.0)
        """
        from bartorch.nlop._newton import _Cell
        from bartorch.nlop.mri import NonlinearSense

        if self.inner is not None or self.alpha_min0:
            raise ValueError(
                "the operator is BART's first form with its own conjugate gradients: it takes "
                "no inner solver and no alpha_min0"
            )
        if 1 > self.iterations:
            raise ValueError("a Gauss-Newton operator takes at least one step")
        if 1 > int(batch):
            raise ValueError("a batch is at least one")
        if not isinstance(F, NonlinearSense):
            from bartorch.nlop.step import Step

            if 1 != int(batch):
                raise ValueError(
                    "only BART's noir model carries a batch of its own; a model assembled "
                    "here takes whatever axes it was built with"
                )
            return Step(F, self, cg_lambda=cg_lambda)
        beyond = [
            name
            for name, asked in (
                ("oversampling_coils", 1.0 != F.oversampling_coils),
                ("oversampled_coils", F.oversampled_coils),
                ("optimized", F.optimized),
                ("coefficient_shape", F.coefficient_shape != F.coil_shape),
            )
            if asked
        ]
        if beyond:
            raise ValueError(
                f"BART's network model fits the coils on the image's grid and cannot be given "
                f"{', '.join(beyond)}; build the NonlinearSense with oversampling_coils=1.0 "
                "and without the others"
            )
        return _Cell(F, self, batch=batch, cg_lambda=cg_lambda)

    # --- BART's second form, with the inner problem anywhere ---------------

    def _in_python(self, y, op, x, ref) -> torch.Tensor:
        """``irgnm2``, written out.

        Every line below is one of ``italgos.c``'s, in its order.  The
        arithmetic is a scale and an add, which ``vecops.c`` does with the
        same kernel torch does, so the bits are the library's -- that is what
        ``inner=optim.CG()`` against :meth:`_in_library` checks.
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


def irgnm(y: torch.Tensor, F, *, x0=None, xref=None, inner=None, **settings):
    """Gauss-Newton for a nonlinear ``F``.  See :class:`IRGNM`."""
    return IRGNM(inner=inner, **settings)(y, F, x0=x0, xref=xref)
