"""The nonlinear operator every concrete one is.

BART's nonlinear operator carries a derivative and the adjoint of that
derivative at whatever point it was last evaluated, which is exactly torch's
forward- and reverse-mode products.  That correspondence is what lets a signal
model written in either place be fitted by a solver written in the other, and
it is the part sigpy has no counterpart for.
"""

from __future__ import annotations

import abc

import torch

from bartorch._lib import library
from bartorch._operator import Built, Operator, Shape, as_operand
from bartorch.core.graph import BartError, _lock, _on_device

__all__ = ["BartNonlinearOperator", "Chain", "FromLinear", "NonlinearOperator"]


class NonlinearOperator(abc.ABC):
    """A map between two C-order shapes, with a derivative and its adjoint.

    The derivative is taken at the point the operator was last evaluated at,
    which is how BART's solvers use one: a forward call fixes the
    linearisation, and the two derivative calls are taken there until the next
    forward call.

    Attributes
    ----------
    ishape, oshape : tuple of int
        Domain and codomain, C order.
    """

    ishape: Shape
    oshape: Shape
    device: torch.device | None = None

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``F(x)``, which also fixes where the derivative is taken."""

    @abc.abstractmethod
    def derivative(self, dx: torch.Tensor) -> torch.Tensor:
        """``DF|x dx`` at the last evaluated point."""

    @abc.abstractmethod
    def adjoint(self, dy: torch.Tensor) -> torch.Tensor:
        """``DF|x^H dy`` at the last evaluated point."""

    def as_bart(self) -> BartNonlinearOperator:
        """An equivalent operator that a BART handle stands behind."""
        from bartorch.nlop.callback import Callback

        return Callback(self.oshape, self.ishape, self.forward, self.derivative, self.adjoint)

    def linearize(self, x: torch.Tensor):
        """The derivative at *x*, as a :class:`~bartorch.linop.LinearOperator`.

        Evaluates the operator at *x* first, which is what fixes the point.
        """
        from bartorch.linop.basic import Callback as LinearCallback

        self.forward(x)
        return LinearCallback(self.oshape, self.ishape, self.derivative, self.adjoint)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """``F(x)``, recorded for autograd when the input asks for it."""
        if isinstance(x, torch.Tensor) and x.requires_grad and torch.is_grad_enabled():
            from bartorch.nlop.autograd import apply

            return apply(self, x)
        return self.forward(x)

    def __matmul__(self, other):
        """``self @ other`` applies ``other`` first."""
        from bartorch.linop.base import LinearOperator

        if isinstance(other, LinearOperator):
            other = other.to_nonlinear()
        if not isinstance(other, NonlinearOperator):
            return NotImplemented
        return Chain(self, other)

    def __rmatmul__(self, other):
        from bartorch.linop.base import LinearOperator

        if not isinstance(other, LinearOperator):
            return NotImplemented
        return Chain(other.to_nonlinear(), self)

    def irgnm(
        self,
        y: torch.Tensor,
        x0: torch.Tensor,
        iterations: int = 8,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        redu: float = 2.0,
        cgiter: int = 30,
        cgtol: float = 0.0,
        xref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit ``self(x) = y`` from ``x0`` by BART's iteratively regularised Gauss-Newton.

        Parameters
        ----------
        y : tensor
            Data of shape :attr:`oshape`.
        x0 : tensor
            Starting point of shape :attr:`ishape`; also the regularisation
            centre unless ``xref`` is given.
        iterations : int
            Gauss-Newton steps.
        alpha, alpha_min, redu : float
            Initial regularisation weight, its floor, and the factor it is
            divided by after each step.
        cgiter, cgtol : int, float
            Conjugate-gradient budget and tolerance for each linearised step.
        xref : tensor, optional
            The regularisation centre, when it is not ``x0``.
        """
        return self.as_bart().irgnm(y, x0, iterations, alpha, alpha_min, redu, cgiter, cgtol, xref)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ishape} -> {self.oshape})"


class BartNonlinearOperator(NonlinearOperator, Operator):
    """A nonlinear operator that one of BART's own handles stands behind."""

    _free_name = "bartorch_nlop_free"
    _domain_name = "bartorch_nlop_domain"
    _codomain_name = "bartorch_nlop_codomain"

    def as_bart(self) -> BartNonlinearOperator:
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(library().bartorch_nlop_apply, x, self.ishape, self.oshape)

    def derivative(self, dx: torch.Tensor) -> torch.Tensor:
        return self._apply(library().bartorch_nlop_derivative, dx, self.ishape, self.oshape)

    def adjoint(self, dy: torch.Tensor) -> torch.Tensor:
        return self._apply(library().bartorch_nlop_adjoint, dy, self.oshape, self.ishape)

    def irgnm(
        self,
        y: torch.Tensor,
        x0: torch.Tensor,
        iterations: int = 8,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        redu: float = 2.0,
        cgiter: int = 30,
        cgtol: float = 0.0,
        xref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = as_operand(y, self.oshape, "y")
        x = as_operand(x0, self.ishape, "x0").clone()
        ref = as_operand(xref, self.ishape, "xref") if xref is not None else None
        with _lock, _on_device(self.device or y.device):
            code = library().bartorch_irgnm(
                self._h.ptr,
                int(iterations),
                float(alpha),
                float(alpha_min),
                float(redu),
                int(cgiter),
                float(cgtol),
                x.data_ptr(),
                y.data_ptr(),
                ref.data_ptr() if ref is not None else None,
            )
        if code != 0:
            raise BartError("Gauss-Newton solve failed; see the log for BART's message")
        return x

    irgnm.__doc__ = NonlinearOperator.irgnm.__doc__


class FromLinear(BartNonlinearOperator):
    """A linear operator seen as a nonlinear one, whose derivative is itself."""

    def __init__(self, op):
        self.op = op.as_bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_from_linop, self.op._h.ptr, device=self.op.device
        )
        return Built(ptr, self.op.ishape, self.op.oshape, keep=(self.op,), device=self.op.device)

    def __repr__(self) -> str:
        return f"{self.op!r}.to_nonlinear()"


class Chain(BartNonlinearOperator):
    """``a @ b``: *b* applied first, as one BART operator."""

    def __init__(self, a: NonlinearOperator, b: NonlinearOperator):
        self.a, self.b = a.as_bart(), b.as_bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_chain,
            self.b._h.ptr,
            self.a._h.ptr,
            device=self.a.device or self.b.device,
        )
        return Built(
            ptr,
            self.b.ishape,
            self.a.oshape,
            keep=(self.a, self.b),
            device=self.a.device or self.b.device,
        )

    def __repr__(self) -> str:
        return f"({self.a!r} @ {self.b!r})"
