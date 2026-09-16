"""Nonlinear inverse problems by iteratively regularized Gauss-Newton.

The method comes in two forms, and :class:`IRGNMBlock` takes one step of
either.  ``irgnm`` solves each linearized problem by conjugate gradients inside
the library, as ``noir``'s step does, which is the block without an inner
solver.  ``irgnm2`` hands the problem to a generic regularized least-squares
solver, which is what ``inner=`` selects.  :class:`IRGNM` loops the block to
BART's schedule, as the solvers in :mod:`bartorch.optim` loop theirs.
"""

from __future__ import annotations

import copy
import dataclasses
import weakref

import torch
from torch import nn

from bartorch._dispatch import BartError, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand
from bartorch.optim.blocks import _setting, _value

__all__ = ["IRGNM", "IRGNMBlock", "irgnm"]


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
    normal)`` and rescales at every step -- so neither does this.
    """
    step = copy.copy(solver)
    step.cclambda = float(alpha)
    if hasattr(step, "eigen"):
        step.eigen = True
    return step


class IRGNMBlock(nn.Module):
    """One step of iteratively regularized Gauss-Newton, ``noir``'s Gauss-Newton step.

    Without ``inner`` the step is BART's first form,

    ``x = xn + (DF^H DF + alpha)^-1 [DF^H (y - F(xn)) - alpha (xn - xref)]``,

    with the inverse computed by conjugate gradients inside the library and
    differentiated implicitly.  With ``inner`` it is the second form: the
    solver minimizes ``||DF u - r||^2 + alpha ||u||^2 + R(u)`` for ``DF``
    linearized at ``xn`` and ``r = y - F(xn) + DF (xn - xref)``, and
    ``x = u + xref``.  The weight then decays as
    ``alpha <- (alpha - alpha_min) / redu + alpha_min``, and in the second form
    is kept above ``alpha_min0``.

    ``state = block.start(y, F, x0, xref)`` prepares the data and the model,
    ``state = block(state, F)`` takes one step, and ``block.output(state, F)``
    returns the unknowns.  ``F`` needs a :attr:`~bartorch.nlop.NonlinearOperator.bundle`.
    A leading batch axis on ``y`` is a batch of independent items, each stepped
    on its own.

    Parameters
    ----------
    alpha : float
        Initial Tikhonov weight.
    alpha_min : float
        What the weight decays towards.
    alpha_min0 : float
        Floor the decayed weight is never taken below; second form only.
    redu : float
        Factor the weight is divided by after each step.
    cg_maxiter, cg_tol, cg_lambda : int, float, float
        The first form's conjugate gradients: ``iter_conjgrad_conf``'s
        ``maxiter``, ``tol`` and ``l2lambda``.  A nonzero ``tol`` is refused
        by BART once the step is differentiated.
    inner : solver, optional
        A configured solver from :mod:`bartorch.optim` for the linearized
        problem; its regularizers are ``R``.
    fuse : bool
        Lower a coil composition so that its encoding is applied once as its
        normal operator; see :meth:`plan`.

    Notes
    -----
    ``alpha`` and ``redu`` are :class:`torch.nn.Parameter` objects, frozen until
    ``requires_grad_()``.  The first form is differentiable by the data, the
    iterate, the centre and ``alpha``.  The second form is differentiable by
    the data, the iterate, the centre and the inner solver's own settings and
    priors; it gives ``alpha`` to the solver as a number, so ``alpha`` is held
    fixed there.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        x: torch.Tensor
        xref: torch.Tensor
        data: torch.Tensor
        alpha: object
        space: object
        k: int = 0

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        alpha_min0: float = 0.0,
        redu: float = 2.0,
        cg_maxiter: int = 30,
        cg_tol: float = 0.0,
        cg_lambda: float = 0.0,
        inner=None,
        fuse: bool = True,
    ):
        super().__init__()
        self.inner = None if inner is None else _inner_solver(inner)
        if self.inner is None and alpha_min0:
            raise ValueError(
                "alpha_min0 is the second form's floor; BART's first form has none, so it "
                "takes an inner solver"
            )
        self.alpha = _setting(alpha)
        self.redu = _setting(redu)
        self.alpha_min = float(alpha_min)
        self.alpha_min0 = float(alpha_min0)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_tol = float(cg_tol)
        self.cg_lambda = float(cg_lambda)
        self.fuse = bool(fuse)
        self._spaces = weakref.WeakKeyDictionary()

    def _space(self, F):
        """``F`` prepared for steps, built once per model."""
        from bartorch.linop.base import LinearOperator
        from bartorch.nlop.step import Linearized

        made = self._spaces.get(F)
        if made is None:
            model = F.to_nonlinear() if isinstance(F, LinearOperator) else F
            made = Linearized(
                model,
                cg_maxiter=self.cg_maxiter,
                cg_tol=self.cg_tol,
                cg_lambda=self.cg_lambda,
                fuse=self.fuse,
                inverse=self.inner is None,
            )
            self._spaces[F] = made
        return made

    def plan(self, F):
        """What ``F`` was lowered into, as a :class:`~bartorch.nlop.plan.Plan`.

        ``plan.bundle`` says where the derivative came from, ``plan.domain``
        whether the step works against the normal operator or applies the
        encoding forward and adjoint, ``plan.encoding`` the linear part's own
        plan, and ``plan.fused`` whether the coil model was rewritten.
        """
        return self._space(F).plan

    def start(self, y: torch.Tensor, F, x0=None, xref=None) -> State:
        """The run's state: the prepared data, the start, the centre and the weight.

        ``x0`` and ``xref`` are a state (the unknowns laid end to end), the one
        unknown of a single-input model at its shape, or a tuple of unknowns.
        Without ``x0`` the start is ``nlinv``'s: the first unknown ones, the
        rest zero.  Without ``xref`` the steps are regularized towards zero.
        """
        space = self._space(F)
        data = space.prepare(torch.as_tensor(y))
        batch = data.shape[:1] if space.batched(data, space.data_shape) else ()
        if x0 is None:
            x = space.start(batch, device=data.device)
        else:
            x = space.state(x0).expand(*batch, *space.state_shape)
        centre = torch.zeros_like(x) if xref is None else space.state(xref).expand_as(x)
        if self.inner is not None:
            alpha = float(self.alpha)
        elif self.alpha.requires_grad:
            alpha = self.alpha.float().to(torch.complex64) * torch.ones_like(x)
        else:
            alpha = torch.full_like(x, float(self.alpha))
        return self.State(x, centre, data, alpha, space)

    def forward(self, state: State, F) -> State:
        space = state.space
        take = self._first if self.inner is None else self._second
        if space.batched(state.x, space.state_shape):
            items = zip(state.x, state.xref, state.data, _items(state.alpha, len(state.x)))
            x = torch.stack([take(space, *item) for item in items])
        else:
            x = take(space, state.x, state.xref, state.data, state.alpha)
        return dataclasses.replace(state, x=x, alpha=self._decay(state.alpha), k=state.k + 1)

    def output(self, state: State, F):
        """The unknowns: a tensor for a model of one input, else one per input."""
        parts = state.space.split(state.x)
        return parts[0] if 1 == len(parts) else parts

    def _first(self, space, x, xref, data, alpha):
        """``noir_gauss_newton_step_create_s``'s expression, one item."""
        from bartorch.nlop.derivative import _evaluate

        residual = data - _evaluate(space.operator, x)
        rhs = _evaluate(space.adjoint, residual, x) - alpha * (x - xref)
        return x + _evaluate(space.inverse(x.device), rhs, x, alpha)

    def _second(self, space, x, xref, data, alpha):
        """``irgnm2``'s step, one item: the linearization carried to the centre."""
        from bartorch.nlop.derivative import _evaluate

        residual = data - _evaluate(space.operator, x)
        derivative = space.flat.at(x)
        residual = residual + derivative.forward(x - xref)
        return _at(self.inner, alpha)(residual, derivative) + xref

    def _decay(self, alpha):
        if self.inner is None:
            redu = _value(self.redu)
            return (alpha - self.alpha_min) / redu + self.alpha_min
        alpha = (alpha - self.alpha_min) / float(self.redu) + self.alpha_min
        return self.alpha_min0 if alpha < self.alpha_min0 else alpha

    def __repr__(self) -> str:
        inner = "" if self.inner is None else f", inner={type(self.inner).__name__}(...)"
        return f"IRGNMBlock(alpha={float(self.alpha)}, redu={float(self.redu)}{inner})"


def _items(alpha, count: int):
    """The weight per item: a batched tensor's rows, or one number for all."""
    return alpha if isinstance(alpha, torch.Tensor) else [alpha] * count


class IRGNM:
    """Iteratively regularized Gauss-Newton for ``F(x) = y``, looping :class:`IRGNMBlock`.

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
        BART's first form, with the inverse inside the library, as ``nlinv``
        does.
    fuse : bool
        Lower a coil composition so that its encoding is applied once as its
        normal operator.

    Examples
    --------
    Plain, which is ``nlinv``:

    >>> nlop.IRGNM(iterations=8)(kspace, F, x0=start)

    The same method with the linearized problem written out here, which is
    ``iter4_irgnm2`` to the bit:

    >>> nlop.IRGNM(iterations=8, inner=optim.CG())(kspace, F, x0=start)

    Wavelet-regularized:

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
    reproduce one without it.  Each reproduces BART's own loop of its form --
    ``iter4_irgnm`` and, with ``inner=optim.CG()``, ``iter4_irgnm2`` -- and the
    suite holds both to that.
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
        fuse: bool = True,
    ):
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_min0 = float(alpha_min0)
        self.redu = float(redu)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_tol = float(cg_tol)
        self.inner = None if inner is None else _inner_solver(inner)
        self.fuse = bool(fuse)

    def _block(self) -> IRGNMBlock:
        """The step this solver loops over, built from its settings."""
        return IRGNMBlock(
            alpha=self.alpha,
            alpha_min=self.alpha_min,
            alpha_min0=self.alpha_min0 if self.inner is not None else 0.0,
            redu=self.redu,
            cg_maxiter=self.cg_maxiter,
            cg_tol=self.cg_tol,
            inner=self.inner,
            fuse=self.fuse,
        )

    def __call__(self, y: torch.Tensor, F, x0=None, xref=None):
        """Fit ``F(x) = y``.

        Parameters
        ----------
        y : torch.Tensor
            Data of ``F.oshape``, with an optional batch axis in front.
        F : NonlinearOperator or LinearOperator
            The forward model, with a bundle; a linear one is converted with
            :meth:`~bartorch.linop.LinearOperator.to_nonlinear`.
        x0 : torch.Tensor or tuple of torch.Tensor, optional
            Starting point; see :meth:`IRGNMBlock.start`.
        xref : torch.Tensor or tuple of torch.Tensor, optional
            Regularization centre.  Without one the steps are regularized
            towards zero, as BART does.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Complex64 solution, one tensor per input of ``F``.
        """
        block = self._block()
        state = block.start(y, F, x0, xref)
        for _ in range(self.iterations):
            state = block(state, F)
        return block.output(state, F)

    # --- BART's own loops, the references the block is held against ---------

    def _first_in_library(
        self, y: torch.Tensor, F, x0: torch.Tensor, xref: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``iter4_irgnm``, BART's first form, run inside the library: the reference
        the block without an inner solver is held against."""
        from bartorch.linop.base import LinearOperator

        if isinstance(F, LinearOperator):
            F = F.to_nonlinear()
        op = F._bart()
        y = as_operand(y, op.oshape, "y")
        x = as_operand(x0, op.ishape, "x0").clone()
        ref = as_operand(xref, op.ishape, "xref") if xref is not None else None
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

    def __repr__(self) -> str:
        inner = "" if self.inner is None else f", inner={type(self.inner).__name__}(...)"
        return (
            f"IRGNM(iterations={self.iterations}, alpha={self.alpha}, "
            f"alpha_min={self.alpha_min}, redu={self.redu}{inner})"
        )


def irgnm(y: torch.Tensor, F, *, x0=None, xref=None, inner=None, **settings):
    """Solve ``F(x) = y`` by iteratively regularized Gauss-Newton.

    Linearizes ``F`` at the current iterate and solves the linearized problem
    under a Tikhonov weight that decays from ``alpha`` towards ``alpha_min``,
    so early steps are strongly regularized and later ones are not.

    Parameters
    ----------
    y : tensor
        Data of ``F.oshapes[0]``.
    F : NonlinearOperator
        The forward model.
    x0 : tensor or tuple of tensor, optional
        Starting iterate; without one the model's own initial value is used.
    xref : tensor or tuple of tensor, optional
        Regularization centre the steps are pulled towards; without one they
        are regularized towards zero.
    inner : solver, optional
        Solver for the linearized problem, from :mod:`bartorch.optim`.  Without
        one it is solved by conjugate gradients inside the library.
    **settings
        Settings of :class:`IRGNM`, among them ``iterations`` (8), ``alpha``,
        ``alpha_min`` and ``redu``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The solution, one tensor per input of ``F``.
    """
    return IRGNM(inner=inner, **settings)(y, F, x0=x0, xref=xref)
