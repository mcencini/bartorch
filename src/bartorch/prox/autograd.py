"""A term's transform as a torch autograd function; the backward pass is its transpose.

The transform is the linear map BART puts in front of a term's proximal
operator -- a wavelet transform, a finite difference, the identity.  An
alternating-direction or primal-dual step applies it to the iterate, so an
unrolled network differentiates through it, and the three modes are each
other's transposes: ``forward`` and ``adjoint`` swap, and ``normal`` is its
own.

The proximal operator itself is not here.  BART's are ``operator_p_s``, which
carries no derivative -- ``operator_p_fun_t`` is ``(data, mu, dst, src)`` and
there is nowhere for one to live -- so a term's threshold is a constant in the
graph.  That is what the denoiser slot is for: whatever stands where a term
goes differentiates on its own terms, and the iteration around it is recorded.
"""

from __future__ import annotations

import torch

__all__ = ["apply_transform"]

#: What each mode's backward pass applies.
_TRANSPOSE = {"forward": "adjoint", "adjoint": "forward", "normal": "normal"}


class _Transform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, term, image_shape, mode):  # noqa: D102
        ctx.term = term
        ctx.image_shape = image_shape
        ctx.mode = mode
        ctx.real = not x.is_complex()
        ctx.dtype = x.dtype
        with torch.no_grad():
            return term._transform_apply(x, image_shape, mode)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        g = ctx.term._transform_apply(
            grad.resolve_conj().contiguous(), ctx.image_shape, _TRANSPOSE[ctx.mode]
        )
        return (g.real.to(ctx.dtype) if ctx.real else g.to(ctx.dtype)), None, None, None


def apply_transform(term, x: torch.Tensor, image_shape, mode: str) -> torch.Tensor:
    """The transform applied, recorded so that the backward pass is its transpose."""
    return _Transform.apply(x, term, image_shape, mode)
