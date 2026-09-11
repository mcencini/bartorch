"""Nonlinear operators as torch autograd functions; the backward pass is the adjoint derivative.

The derivative is the one fixed by the last forward evaluation, so the operator
must not be evaluated elsewhere between a forward pass and its backward pass,
and is not safe to share between graphs evaluated concurrently.
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
