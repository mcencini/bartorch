"""The derivative of one output of a nonlinear operator by one of its inputs."""

from __future__ import annotations

from bartorch._lib import library
from bartorch._operator import Built
from bartorch.linop.base import LinearOperator

__all__ = ["Derivative"]


class Derivative(LinearOperator):
    """``DF/dx_input`` of one output of a nonlinear operator, as a linear one.

    ``nlop_get_derivative``.  This is a view: the point is wherever the
    operator's last application left it, and it moves with the next one.  That
    is what makes it usable as the inner problem of a Gauss-Newton step, where
    the linearisation point is the iterate, and what makes it wrong to hold on
    to across an unrelated evaluation.

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
