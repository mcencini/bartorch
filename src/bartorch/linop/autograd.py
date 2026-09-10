"""An operator as a torch function, whose backward pass is its adjoint.

For a linear map that is not an approximation but the definition.  Torch
stores conjugate Wirtinger gradients: for ``y = A x`` what a backward pass has
to hand back is ``A^H g``, which is the adjoint and not the transpose.  The
near miss -- returning the transpose -- is silent, and wrong by a conjugation
that a real-valued test would never notice.

A real input is complexified on the way in and its gradient is taken real on
the way out, which is the derivative of the embedding of the reals in the
complex numbers and is what makes a real image parametrization work.

What is *not* differentiated is the operator itself.  The sensitivities and
the trajectory are baked into a BART handle, so a gradient with respect to
them -- autofocus, motion, B0 -- is not available here; an operator that grows
one adds it to its own function rather than to this one.
"""

from __future__ import annotations

import torch

__all__ = ["apply_adjoint", "apply_forward"]


def _restore(grad: torch.Tensor, real: bool, dtype: torch.dtype) -> torch.Tensor:
    """A gradient in the domain the input came from."""
    return grad.real.to(dtype) if real else grad.to(dtype)


class _Forward(torch.autograd.Function):
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
        return _restore(g, ctx.real, ctx.dtype), None


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, op):  # noqa: D102
        ctx.op = op
        ctx.real = not y.is_complex()
        ctx.dtype = y.dtype
        with torch.no_grad():
            return op.adjoint(y)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        g = ctx.op.forward(grad.resolve_conj().contiguous())
        return _restore(g, ctx.real, ctx.dtype), None


def apply_forward(op, x: torch.Tensor) -> torch.Tensor:
    """``A x``, recorded so that the backward pass is ``A^H``."""
    return _Forward.apply(x, op)


def apply_adjoint(op, y: torch.Tensor) -> torch.Tensor:
    """``A^H y``, recorded so that the backward pass is ``A``."""
    return _Adjoint.apply(y, op)
