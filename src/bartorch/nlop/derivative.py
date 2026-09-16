"""The derivative of one output of a nonlinear operator by one of its inputs."""

from __future__ import annotations

from bartorch._lib import library
from bartorch._operator import Built
from bartorch.linop.base import LinearOperator

__all__ = ["Derivative"]


class Derivative(LinearOperator):
    """Derivative of one output by one input at the last evaluated point, as a linear operator.

    ``nlop_get_derivative``.  This is a view: the point is wherever the
    operator's last application left it, and it moves with the next one, so
    it is wrong to hold on to across an unrelated evaluation.
    :meth:`~bartorch.nlop.NonlinearOperator.linearize` gives the derivative at a point it holds.

    Parameters
    ----------
    op : NonlinearOperator
        A BART-backed operator.  One defined in Python by ``forward``,
        ``derivative`` and ``adjoint`` has no handle to take a derivative of;
        use :meth:`~bartorch.nlop.NonlinearOperator.linearize` for those.
    output, input : int
        Which output and which input, counted BART's way.
    """

    def __init__(self, op, output: int = 0, input: int = 0):  # noqa: A002
        if not op._native:
            raise NotImplementedError(
                f"{type(op).__name__} is defined in Python and has no BART operator to take "
                "a derivative of; linearize() wraps its own derivative instead"
            )
        from bartorch.nlop.base import _index

        self.op = op
        self.output = _index(output, len(op.oshapes), "output")
        self.input = _index(input, len(op.ishapes), "input")
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_derivative_linop,
            self.op._h.ptr,
            self.output,
            self.input,
            device=self.op.device,
        )
        return Built(
            ptr,
            self.op.ishapes[self.input],
            self.op.oshapes[self.output],
            keep=(self.op,),
            device=self.op.device,
        )

    def __repr__(self) -> str:
        return f"{self.op!r}.jacobian({self.output}, {self.input})"


def _evaluate(op, *xs):
    """``op(*xs)``, recorded with its point restored before the backward pass."""
    import torch

    from bartorch.linop.base import _tracking
    from bartorch.nlop.autograd import apply

    if any(_tracking(x) for x in xs):
        return apply(op, *xs, restore=True)
    with torch.no_grad():
        return op.forward(*xs)


def _written(value, out):
    if out is None:
        return value
    out.copy_(value)
    return out


class Linearization(LinearOperator):
    """Derivative ``DF(xn)`` of a nonlinear operator, with ``xn`` held as a tensor.

    Each application passes the point to the bundle, so it does not depend on
    what was evaluated before it, and is differentiable by the point as well as
    by its argument.  A bundle lowered into the normal-equation domain gives an
    asymmetric pair: ``adjoint`` is ``DF^H`` and ``forward`` is ``E^H E DF``.
    """

    #: Applications record themselves; the linear autograd wrappers step aside.
    _records = True

    def __init__(self, bundle, point):
        op = bundle.operator
        if 1 != len(op.ishapes) or 1 != len(op.oshapes):
            raise ValueError(
                "a linearization is of an operator with one input and one output; flatten "
                f"{type(op).__name__} first"
            )
        self._members, self.point = bundle, point
        self.ishape, self.oshape = tuple(op.ishapes[0]), tuple(op.oshapes[0])
        self.device = getattr(point, "device", None)
        super().__init__()

    def at(self, point) -> Linearization:
        """The same derivative at another point."""
        return Linearization(self._members, point)

    def forward(self, x, out=None):
        return _written(_evaluate(self._members.derivative, x, self.point), out)

    def adjoint(self, y, out=None):
        return _written(_evaluate(self._members.adjoint, y, self.point), out)

    def normal(self, x, out=None):
        return _written(_evaluate(self._members.normal, x, self.point), out)

    def _as_callbacks(self) -> LinearOperator:
        import torch

        def plain(apply):
            def run(v):
                with torch.no_grad():
                    return apply(v)

            return run

        return LinearOperator.from_callbacks(
            self.oshape, self.ishape, plain(self.forward), plain(self.adjoint), plain(self.normal)
        )

    def __repr__(self) -> str:
        return f"Linearization({type(self._members.operator).__name__})"
