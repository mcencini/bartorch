"""Linear operators as torch autograd functions; the backward pass is the adjoint.

For complex tensors that is the conjugate Wirtinger gradient torch expects.  A
real input is treated as embedded in the complex numbers, and its gradient is
the real part.  Gradients with respect to the operator's own data (sensitivities,
trajectory) are not computed, except by an operator that records its own
applications (``_records``), which these wrappers hand the argument to.
"""

from __future__ import annotations

import torch

__all__ = ["apply_adjoint", "apply_forward", "apply_normal"]


def _restore(grad: torch.Tensor, real: bool, dtype: torch.dtype) -> torch.Tensor:
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
    if getattr(op, "_records", False):
        return op.forward(x)
    return _Forward.apply(x, op)


def apply_adjoint(op, y: torch.Tensor) -> torch.Tensor:
    """``A^H y``, recorded so that the backward pass is ``A``."""
    if getattr(op, "_records", False):
        return op.adjoint(y)
    return _Adjoint.apply(y, op)


class _NormalOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op):  # noqa: D102
        ctx.op = op
        ctx.real = not x.is_complex()
        ctx.dtype = x.dtype
        with torch.no_grad():
            return op.normal(x)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        # A^H A is its own adjoint, so the backward pass is the same operator.
        g = ctx.op.normal(grad.resolve_conj().contiguous())
        return _restore(g, ctx.real, ctx.dtype), None


def apply_normal(op, x: torch.Tensor) -> torch.Tensor:
    """``A^H A x``, recorded so that the backward pass is ``A^H A`` again."""
    if getattr(op, "_records", False):
        return op.normal(x)
    return _NormalOp.apply(x, op)
