"""A nonlinear operator as a torch function.

The backward pass is the adjoint of the derivative at the point the forward
pass evaluated, which is what BART's nonlinear operator carries and what a
vector-Jacobian product is.  The forward pass has to be the one that fixed
that point, so nothing may evaluate the operator between the two -- which is
true of a single backward pass and is why the operator is not safe to share
across two graphs evaluated in parallel.
"""

from __future__ import annotations

import torch

__all__ = ["apply"]


class _Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op):  # noqa: D102
        ctx.op = op
        ctx.real = not x.is_complex()
        ctx.dtype = x.dtype
        with torch.no_grad():
            return op.forward(x)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        g = ctx.op.adjoint(grad.resolve_conj().contiguous())
        if ctx.real:
            g = g.real
        return g.to(ctx.dtype), None


def apply(op, x: torch.Tensor) -> torch.Tensor:
    """``F(x)``, recorded so that the backward pass is ``DF|x^H``."""
    return _Apply.apply(x, op)
