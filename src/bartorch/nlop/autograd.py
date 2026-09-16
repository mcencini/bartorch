"""Nonlinear operators as torch autograd functions; the backward pass is the adjoint derivative.

The derivative is the one fixed by the last forward evaluation.  Without
``restore`` the operator must not be evaluated elsewhere between a forward pass
and its backward pass; with it, the backward pass evaluates the operator at the
saved inputs again first, as ``nlop_checkpoint`` does.  Neither is safe to share
between graphs evaluated concurrently.
"""

from __future__ import annotations

import torch

__all__ = ["apply"]


class _Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, restore, *xs):  # noqa: D102
        ctx.op = op
        ctx.restore = restore
        if restore:
            ctx.save_for_backward(*xs)
        ctx.real = tuple(isinstance(x, torch.Tensor) and not x.is_complex() for x in xs)
        ctx.dtypes = tuple(x.dtype if isinstance(x, torch.Tensor) else None for x in xs)
        ctx.shapes = tuple(x.shape if isinstance(x, torch.Tensor) else None for x in xs)
        ctx.device = next((x.device for x in xs if isinstance(x, torch.Tensor)), None)
        with torch.no_grad():
            return op.forward(*xs)

    @staticmethod
    def backward(ctx, *grads):  # noqa: D102
        op = ctx.op
        if ctx.restore:
            with torch.no_grad():
                op.forward(*ctx.saved_tensors)
        grads = [g.resolve_conj().contiguous() if g is not None else None for g in grads]
        out = [None, None]
        for at, (real, dtype, shape) in enumerate(zip(ctx.real, ctx.dtypes, ctx.shapes)):
            if not ctx.needs_input_grad[at + 2]:
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
                    g = torch.zeros(op.ishapes[at], dtype=torch.complex64, device=ctx.device)
            if real:
                g = g.real
            # The operator took the input at whatever shape held its entries;
            # the gradient goes back at that shape.
            out.append(g.to(dtype).reshape(shape))
        return tuple(out)


def apply(op, *xs: torch.Tensor, restore: bool = False):
    """``F(x)``, recorded so that the backward pass is ``DF|x^H``.

    ``restore`` evaluates ``F`` at ``x`` again before the backward pass, for an
    operator applied at other points in between.
    """
    return _Apply.apply(op, restore, *xs)
