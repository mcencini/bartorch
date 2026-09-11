"""The nonlinear operator base class and composition."""

from __future__ import annotations

import torch

from bartorch._lib import library
from bartorch._operator import Built, Operator

__all__ = ["Chain", "FromLinear", "NonlinearOperator"]


class NonlinearOperator(Operator):
    """A map between two C-order shapes, with a derivative and its adjoint.

    :meth:`derivative` and :meth:`adjoint` are taken at the point of the last
    :meth:`forward` call, which is how BART's solvers use them.  The backward
    pass of ``F(x)`` is ``adjoint`` at that point, so evaluating the operator
    elsewhere between a forward and a backward pass gives a wrong gradient.

    A subclass is defined either by :meth:`_create`, which builds one of
    BART's operators, or in Python by :meth:`forward`, :meth:`derivative` and
    :meth:`adjoint`.

    Attributes
    ----------
    ishape, oshape : tuple of int
        Domain and codomain, C order.
    """

    _free_name = "bartorch_nlop_free"
    _domain_name = "bartorch_nlop_domain"
    _codomain_name = "bartorch_nlop_codomain"

    def __init__(self):
        if self._native:
            self._build()
        elif any(
            getattr(type(self), name) is getattr(NonlinearOperator, name)
            for name in ("forward", "derivative", "adjoint")
        ):
            raise TypeError(
                f"{type(self).__name__} must define _create, or forward, derivative and adjoint"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``F(x)``, which also fixes where the derivative is taken."""
        return self._apply(library().bartorch_nlop_apply, x, self.ishape, self.oshape)

    def derivative(self, dx: torch.Tensor) -> torch.Tensor:
        """``DF(x) dx`` at the last evaluated point."""
        return self._apply(library().bartorch_nlop_derivative, dx, self.ishape, self.oshape)

    def adjoint(self, dy: torch.Tensor) -> torch.Tensor:
        """``DF(x)^H dy`` at the last evaluated point."""
        return self._apply(library().bartorch_nlop_adjoint, dy, self.oshape, self.ishape)

    def _as_callbacks(self) -> NonlinearOperator:
        from bartorch.nlop.callback import Callback

        return Callback(self.oshape, self.ishape, self.forward, self.derivative, self.adjoint)

    def linearize(self, x: torch.Tensor):
        """The derivative at ``x``, as a :class:`~bartorch.linop.LinearOperator`.

        Evaluates the operator at ``x``, which moves the point the derivative
        is taken at.
        """
        from bartorch.linop.basic import Callback as LinearCallback

        self.forward(x)
        return LinearCallback(self.oshape, self.ishape, self.derivative, self.adjoint)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """``F(x)``, recorded for autograd when ``x`` requires a gradient."""
        if isinstance(x, torch.Tensor) and x.requires_grad and torch.is_grad_enabled():
            from bartorch.nlop.autograd import apply

            return apply(self, x)
        return self.forward(x)

    def __matmul__(self, other):
        """``self @ other`` applies ``other`` first, as one BART operator."""
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


class FromLinear(NonlinearOperator):
    """A linear operator as a nonlinear one, whose derivative is itself."""

    def __init__(self, op):
        self.op = op._bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_from_linop, self.op._h.ptr, device=self.op.device
        )
        return Built(ptr, self.op.ishape, self.op.oshape, keep=(self.op,), device=self.op.device)

    def __repr__(self) -> str:
        return f"{self.op!r}.to_nonlinear()"


class Chain(NonlinearOperator):
    """``a @ b`` as one BART operator; ``b`` is applied first."""

    def __init__(self, a: NonlinearOperator, b: NonlinearOperator):
        self.a, self.b = a._bart(), b._bart()
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
