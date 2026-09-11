"""The linear operator base class and operator algebra."""

from __future__ import annotations

import torch

from bartorch._lib import library
from bartorch._operator import Built, Operator

__all__ = ["Add", "Adjoint", "Compose", "LinearOperator"]


def _tracking(x) -> bool:
    return isinstance(x, torch.Tensor) and x.requires_grad and torch.is_grad_enabled()


class LinearOperator(Operator):
    """A linear map between two C-order shapes, with an adjoint.

    A subclass is defined either by :meth:`_create`, which builds one of
    BART's operators, or in Python by :meth:`forward` and :meth:`adjoint`, and
    :meth:`normal` where a cheaper form exists.  BART reaches a Python-defined
    operator through callbacks, one crossing into Python per application.

    Both kinds compose into a single BART operator, are solved by
    :mod:`bartorch.optim`, and differentiate in torch.  The backward pass of
    ``A(x)`` is ``A.adjoint``, which for complex tensors is the conjugate
    Wirtinger gradient torch expects, not the transpose.

    Attributes
    ----------
    ishape, oshape : tuple of int
        Domain and codomain, C order.
    device : torch.device or None
        Where the operator does its arithmetic, when that is not where its
        operands are.
    """

    _free_name = "bartorch_linop_free"
    _domain_name = "bartorch_linop_domain"
    _codomain_name = "bartorch_linop_codomain"

    def __init__(self):
        if self._native:
            self._build()
        elif (
            type(self).forward is LinearOperator.forward
            or type(self).adjoint is LinearOperator.adjoint
        ):
            raise TypeError(f"{type(self).__name__} must define _create, or forward and adjoint")

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A x``, without recording for autograd.

        ``out`` is written in place when given; for a BART-backed operator it
        must be a contiguous complex64 tensor of :attr:`oshape` on the input's
        device.
        """
        return self._apply(library().bartorch_linop_forward, x, self.ishape, self.oshape, out)

    def adjoint(self, y: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A^H y``, without recording for autograd."""
        return self._apply(library().bartorch_linop_adjoint, y, self.oshape, self.ishape, out)

    def normal(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A^H A x``.

        A BART-backed operator applies its own normal, which for a
        non-Cartesian encoding built with ``toeplitz=True`` is a convolution
        with a point spread function.  Otherwise ``adjoint(forward(x))``.
        """
        if self._native:
            return self._apply(library().bartorch_linop_normal, x, self.ishape, self.ishape, out)
        return self.adjoint(self.forward(x), out)

    def _as_callbacks(self) -> LinearOperator:
        from bartorch.linop.basic import Callback

        return Callback(self.oshape, self.ishape, self.forward, self.adjoint, self.normal)

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        """``self @ other`` applies ``other`` first, as one BART operator."""
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return Compose(self, other)

    def __add__(self, other: LinearOperator) -> LinearOperator:
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return Add(self, other)

    @property
    def H(self) -> LinearOperator:  # noqa: N802  (the mathematical name)
        """The adjoint, as an operator."""
        return Adjoint(self)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A x``, recorded for autograd when ``x`` requires a gradient.

        Parameters
        ----------
        x : torch.Tensor
            Array of :attr:`ishape`.
        out : torch.Tensor, optional
            Contiguous complex64 array of :attr:`oshape` to write into, so that
            repeated applications reuse one buffer.  Not allowed when ``x``
            requires a gradient.
        """
        if _tracking(x):
            if out is not None:
                raise ValueError(
                    "out= writes in place, which autograd cannot record; "
                    "drop it, or detach the input"
                )
            from bartorch.linop.autograd import apply_forward

            return apply_forward(self, x)
        return self.forward(x, out)

    def to_nonlinear(self):
        """The same operator as a :class:`~bartorch.nlop.NonlinearOperator`."""
        from bartorch.nlop.base import FromLinear

        return FromLinear(self)

    # deepinv's names for the same operations, so that an operator can stand
    # in for a LinearPhysics without deepinv being imported.

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """``A x``, under ``deepinv``'s name."""
        return self(x)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """``A^H y``, under ``deepinv``'s name, recorded for autograd."""
        if _tracking(y):
            from bartorch.linop.autograd import apply_adjoint

            return apply_adjoint(self, y)
        return self.adjoint(y)

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """Least-squares solution by :class:`bartorch.optim.CG`.

        ``x0`` is the warm start; the other keywords configure the solver.
        """
        from bartorch.optim import CG

        x0 = kwargs.pop("x0", None)
        return CG(**kwargs)(y, self, x0)


class Compose(LinearOperator):
    """``a @ b`` as one BART operator; ``b`` is applied first."""

    def __init__(self, a: LinearOperator, b: LinearOperator):
        self.a, self.b = a._bart(), b._bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_chain,
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


class Add(LinearOperator):
    """``a + b`` as one BART operator; the two must have the same shapes."""

    def __init__(self, a: LinearOperator, b: LinearOperator):
        self.a, self.b = a._bart(), b._bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_plus,
            self.a._h.ptr,
            self.b._h.ptr,
            device=self.a.device or self.b.device,
        )
        return Built(
            ptr,
            self.a.ishape,
            self.a.oshape,
            keep=(self.a, self.b),
            device=self.a.device or self.b.device,
        )

    def __repr__(self) -> str:
        return f"({self.a!r} + {self.b!r})"


class Adjoint(LinearOperator):
    """``A^H`` as an operator.

    Applying it calls ``A.adjoint``.  BART has no adjoint-of-an-operator
    constructor, so composing it wraps it as callbacks.
    """

    def __init__(self, op: LinearOperator):
        self.op = op
        self.ishape, self.oshape = op.oshape, op.ishape
        self.device = op.device

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.op.adjoint(x, out)

    def adjoint(self, y: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.op.forward(y, out)

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        return self.op

    def __repr__(self) -> str:
        return f"{self.op!r}.H"
