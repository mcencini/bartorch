"""The linear operator every concrete one is.

:class:`LinearOperator` is the interface: two shapes, a forward, an adjoint,
and a normal.  Everything written against an operator -- a solver, a torch
model, ``deepinv`` -- should be written against this, and an operator of one's
own is a subclass of it with two methods.

:class:`BartLinearOperator` is one that a BART handle stands behind, which is
what the concrete operators in this package are.  Those compose into BART's
own composite rather than into a Python chain, so a chain of five applies as
one call and a solver iterating on it never returns to Python.

Either kind is a torch function: applying one to a tensor that requires a
gradient records it, and the backward pass is the adjoint.  For a linear map
that is not an approximation but the definition -- torch stores conjugate
Wirtinger gradients, so what a backward pass wants for ``y = A x`` is
``A^H g``, which is the adjoint and not the transpose.
"""

from __future__ import annotations

import abc

import torch

from bartorch._lib import library
from bartorch._operator import Operator, Shape, as_operand
from bartorch.core.graph import BartError, _lock, _on_device

__all__ = ["Add", "Adjoint", "BartLinearOperator", "Compose", "LinearOperator"]


def _tracking(x) -> bool:
    """Whether applying an operator to *x* has to be recorded for autograd."""
    return isinstance(x, torch.Tensor) and x.requires_grad and torch.is_grad_enabled()


class LinearOperator(abc.ABC):
    """A linear map between two C-order shapes, with an adjoint.

    The concrete operators are in :mod:`bartorch.linop`.  An operator of one's
    own is this class with :meth:`forward` and :meth:`adjoint` written -- it
    then chains with BART's, goes to BART's solvers, and differentiates in
    torch like any of them.

    Attributes
    ----------
    ishape, oshape : tuple of int
        Domain and codomain, C order.
    device : torch.device or None
        Where the operator does its arithmetic, when that is not simply where
        its operands are.

    Notes
    -----
    :meth:`A`, :meth:`A_adjoint` and :meth:`A_dagger` are the same three
    operations under the names ``deepinv`` gives them, so an operator can
    stand in for a ``LinearPhysics`` without being one, and
    :meth:`to_deepinv` returns one that is.  ``deepinv`` is not a dependency
    of this package: inheriting from it would make every operator cost that
    import and tie this package's releases to theirs.
    """

    ishape: Shape
    oshape: Shape
    device: torch.device | None = None

    # --- what a concrete operator says ----------------------------------

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A x``, without recording anything for autograd."""

    @abc.abstractmethod
    def adjoint(self, y: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A^H y``, without recording anything for autograd."""

    def normal(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A^H A x``, by whatever route the operator has for it.

        The two applications, unless the operator has something better.  For a
        non-Cartesian encoding it does: one multiply against a point spread
        function rather than a transform each way.
        """
        return self.adjoint(self.forward(x), out)

    def as_bart(self) -> BartLinearOperator:
        """An equivalent operator that a BART handle stands behind.

        BART's solvers and BART's composition need a handle, so one written in
        Python is wrapped as a pair of callbacks -- which is what lets it be
        chained with BART's own and iterated on by BART's, with only the
        callbacks crossing back into Python.
        """
        from bartorch.linop.basic import Callback

        return Callback(self.oshape, self.ishape, self.forward, self.adjoint, self.normal)

    # --- algebra --------------------------------------------------------

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
        """The adjoint, as an operator in its own right."""
        return Adjoint(self)

    # --- application ----------------------------------------------------

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """The operator applied to ``x``, written into ``out`` when one is given.

        A tensor that requires a gradient goes through autograd, so an
        operator drops into a torch model as it stands and its backward pass
        is :meth:`adjoint`.  Everything else takes the direct path, which is
        what a solver applying the operator every iteration wants: passing the
        same ``out`` each time reuses one array rather than asking for a fresh
        one, and a fresh host array costs its pages being faulted in as it is
        written -- which for an image crossing from a card is most of the
        crossing.

        Parameters
        ----------
        x : tensor
            An array of :attr:`ishape`.
        out : tensor, optional
            A contiguous complex64 array of :attr:`oshape` to write into.
            Cannot be combined with a tensor that requires a gradient: an
            array written in place is not one autograd can hold.
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

    # --- solving --------------------------------------------------------

    def lstsq(
        self,
        y: torch.Tensor,
        lambda_: float = 0.0,
        maxiter: int = 30,
        tol: float = 1e-6,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve ``min ||A x - y||^2 + lambda ||x||^2`` by BART's conjugate gradients.

        The iteration runs inside the library.  One of BART's own operators
        never leaves C; one written in Python is called back into once per
        application and no more.

        Parameters
        ----------
        y : tensor
            Data of :attr:`oshape`.
        lambda_ : float
            Tikhonov weight.
        maxiter : int
            Conjugate-gradient steps.
        tol : float
            Residual the iteration stops at.
        x0 : tensor, optional
            Starting point; without one the iteration starts at zero.
        """
        return self.as_bart().lstsq(y, lambda_, maxiter, tol, x0)

    def to_nonlinear(self):
        """The same operator as a :class:`~bartorch.nlop.NonlinearOperator`."""
        from bartorch.nlop.base import FromLinear

        return FromLinear(self.as_bart())

    # --- deepinv ---------------------------------------------------------

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """``A x``, under ``deepinv``'s name for it."""
        return self(x)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """``A^H y``, under ``deepinv``'s name for it."""
        if _tracking(y):
            from bartorch.linop.autograd import apply_adjoint

            return apply_adjoint(self, y)
        return self.adjoint(y)

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """The least-squares pseudo-inverse, under ``deepinv``'s name for it."""
        return self.lstsq(y, **kwargs)

    def to_deepinv(self, **kwargs):
        """This operator as a ``deepinv.physics.LinearPhysics``.

        ``deepinv`` is an optional dependency, imported here rather than by
        the package, so an operator costs nothing to build without it.
        """
        from bartorch.interop.deepinv import as_physics

        return as_physics(self, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ishape} -> {self.oshape})"


class BartLinearOperator(LinearOperator, Operator):
    """A linear operator that one of BART's own handles stands behind.

    Every concrete operator in :mod:`bartorch.linop` is one.  Subclass it to
    reach a BART constructor this package does not wrap yet: record what the
    constructor needs, then answer :meth:`_create` with the call.
    """

    _free_name = "bartorch_linop_free"
    _domain_name = "bartorch_linop_domain"
    _codomain_name = "bartorch_linop_codomain"

    def as_bart(self) -> BartLinearOperator:
        return self

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self._apply(library().bartorch_linop_forward, x, self.ishape, self.oshape, out)

    def adjoint(self, y: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self._apply(library().bartorch_linop_adjoint, y, self.oshape, self.ishape, out)

    def normal(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self._apply(library().bartorch_linop_normal, x, self.ishape, self.ishape, out)

    def lstsq(
        self,
        y: torch.Tensor,
        lambda_: float = 0.0,
        maxiter: int = 30,
        tol: float = 1e-6,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = as_operand(y, self.oshape, "y")
        if x0 is None:
            x = torch.zeros(self.ishape, dtype=torch.complex64, device=y.device)
        else:
            x = as_operand(x0, self.ishape, "x0").clone()
        with _lock, _on_device(self.device or y.device):
            code = library().bartorch_lsqr(
                self._h.ptr,
                int(maxiter),
                float(lambda_),
                float(tol),
                int(x0 is not None),
                x.data_ptr(),
                y.data_ptr(),
            )
        if code != 0:
            raise BartError("least-squares solve failed; see the log for BART's message")
        return x

    lstsq.__doc__ = LinearOperator.lstsq.__doc__


class Compose(BartLinearOperator):
    """``a @ b``: *b* applied first, as one BART operator rather than two.

    Parameters
    ----------
    a, b : LinearOperator
        Applied right to left, as the matrix product reads.  One written in
        Python is wrapped as callbacks so that the composite is still a single
        BART operator.
    """

    def __init__(self, a: LinearOperator, b: LinearOperator):
        self.a, self.b = a.as_bart(), b.as_bart()
        super().__init__()

    def _create(self):
        from bartorch._operator import Built

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


class Add(BartLinearOperator):
    """``a + b``, over the shapes the two share."""

    def __init__(self, a: LinearOperator, b: LinearOperator):
        self.a, self.b = a.as_bart(), b.as_bart()
        super().__init__()

    def _create(self):
        from bartorch._operator import Built

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
    """``A^H`` as an operator of its own.

    BART has no adjoint-of-an-operator constructor, so this is the operator
    read the other way round rather than a second handle: applying it costs
    exactly what ``A.adjoint`` costs.  Only composing it needs a handle, and
    :meth:`as_bart` makes one then -- a pair of callbacks, so a chain
    containing an adjoint crosses into Python once per application.
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
