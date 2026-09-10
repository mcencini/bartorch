"""Nonlinear operators written here, for BART's solvers to drive."""

from __future__ import annotations

from collections.abc import Callable

import torch

from bartorch import _marshal
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, callback, dims
from bartorch.nlop.base import BartNonlinearOperator

__all__ = ["Callback", "FromTorch"]


class Callback(BartNonlinearOperator):
    """A nonlinear operator implemented by Python functions on tensors.

    ``forward(x)`` evaluates the operator and fixes the point at which
    ``derivative(dx)`` and ``adjoint(dy)`` are taken until the next forward
    call, which is how BART's solvers use them.

    Parameters
    ----------
    oshape, ishape : tuple of int
        Codomain and domain shapes, C order.
    forward : callable
        Evaluates the operator, and fixes the linearisation point.
    derivative, adjoint : callable
        The derivative at that point and the adjoint of it.
    """

    def __init__(
        self,
        oshape: Shape,
        ishape: Shape,
        forward: Callable[[torch.Tensor], torch.Tensor],
        derivative: Callable[[torch.Tensor], torch.Tensor],
        adjoint: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        self.forward_fn = forward
        self.derivative_fn = derivative
        self.adjoint_fn = adjoint
        super().__init__()

    def _create(self) -> Built:
        ishape, oshape = self.ishape, self.oshape
        fwd = callback(self.forward_fn, ishape, oshape, "forward")
        der = callback(self.derivative_fn, ishape, oshape, "derivative")
        adj = callback(self.adjoint_fn, oshape, ishape, "adjoint")
        ptr = self._under_lock(
            library().bartorch_nlop_callback,
            DIMS,
            dims(oshape),
            DIMS,
            dims(ishape),
            fwd,
            der,
            adj,
            None,
            _marshal.null_release(),
        )
        keep = (fwd, der, adj, self.forward_fn, self.derivative_fn, self.adjoint_fn)
        return Built(ptr, ishape, oshape, keep=keep)


class FromTorch(Callback):
    """A nonlinear operator from a differentiable torch function.

    The derivative is the forward-mode Jacobian-vector product and its adjoint
    the reverse-mode vector-Jacobian product at the last point the operator was
    evaluated, so a torchsim signal model or any other autograd-differentiable
    map can be fitted by BART's Gauss-Newton solver.

    Parameters
    ----------
    fn : callable
        A differentiable function on tensors.
    ishape, oshape : tuple of int
        Domain and codomain, C order.

    Examples
    --------
    >>> F = FromTorch(lambda p: p[0] * torch.exp(-t / p[1]), (2,), t.shape)
    >>> F.irgnm(measured, x0=torch.tensor([1.0, 20.0]))
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], ishape: Shape, oshape: Shape):
        self.fn = fn
        state: dict[str, torch.Tensor] = {}

        def forward(x: torch.Tensor) -> torch.Tensor:
            xr = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                y = fn(xr)
            state["x"], state["y"] = xr, y
            return y.detach()

        def derivative(dx: torch.Tensor) -> torch.Tensor:
            x = state["x"].detach()
            with torch.enable_grad():
                _, jvp = torch.func.jvp(fn, (x,), (dx.detach().clone(),))
            return jvp

        def adjoint(dy: torch.Tensor) -> torch.Tensor:
            x, y = state["x"], state["y"]
            with torch.enable_grad():
                (g,) = torch.autograd.grad(
                    y, x, grad_outputs=dy.detach().clone(), retain_graph=True
                )
            return g

        super().__init__(oshape, ishape, forward, derivative, adjoint)
