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
    def forward(ctx, op, *xs):  # noqa: D102
        ctx.op = op
        ctx.real = tuple(isinstance(x, torch.Tensor) and not x.is_complex() for x in xs)
        ctx.dtypes = tuple(x.dtype if isinstance(x, torch.Tensor) else None for x in xs)
        with torch.no_grad():
            return op.forward(*xs)

    @staticmethod
    def backward(ctx, *grads):  # noqa: D102
        op = ctx.op
        grads = [g.resolve_conj().contiguous() if g is not None else None for g in grads]
        out = [None]
        for at, (real, dtype) in enumerate(zip(ctx.real, ctx.dtypes)):
            if not ctx.needs_input_grad[at + 1]:
                out.append(None)
                continue
            if 1 == len(op.ishapes) == len(op.oshapes):
                # The cheap path, and the only one a Python-defined operator has.
                g = op.adjoint(grads[0])
            else:
                g = None
                for o, grad in enumerate(grads):
                    if grad is None:
                        continue
                    part = op.jacobian(o, at).adjoint(grad)
                    g = part if g is None else g + part
                if g is None:
                    g = torch.zeros(op.ishapes[at], dtype=torch.complex64)
            if real:
                g = g.real
            out.append(g.to(dtype))
        return tuple(out)


def apply(op, *xs: torch.Tensor):
    """``F(x)``, recorded so that the backward pass is ``DF|x^H``."""
    return _Apply.apply(op, *xs)
