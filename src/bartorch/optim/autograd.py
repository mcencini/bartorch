"""The linear solve as a torch autograd function; the backward pass is another solve.

Conjugate gradients is a loop, and differentiating it by unrolling would record
every iteration and give the derivative of the *truncated iteration* rather
than of the answer.  What is recorded here is the derivative of the solve.
``x = N^-1 A^H y`` with ``N = A^H A + lambda I`` is linear in ``y``, so the
vector-Jacobian product is ``(N^-1 A^H)^H = A N^-1`` -- one more solve with the
same normal operator, and then one forward application.

That is BART's own choice too.  ``src/nlops/norm_inv.c`` is the one solver in
the library that is an ``nlop``, and its derivative and adjoint --
``norm_inv_der_src`` and ``norm_inv_adj_src`` -- each run a conjugate-gradient
solve of their own rather than unrolling the forward one.

Two things follow from differentiating the answer rather than the iteration.
A warm start carries no gradient: the solution of a linear system does not
depend on where the iteration began, so ``x0`` is treated as the constant it
mathematically is.  And the backward solve is only as accurate as its own
iteration count -- a forward solve stopped after ten iterations gets a backward
stopped after ten, and neither is the exact inverse.

Gradients with respect to the operator's own data -- sensitivities, a
trajectory, the weight on a term -- are not computed, which is what
:mod:`bartorch.linop.autograd` says of a plain application as well.  Neither
is a second derivative: the backward pass runs the solve inside the library
and records nothing of its own, so ``create_graph=True`` gets a first-order
gradient with no history behind it.
"""

from __future__ import annotations

import torch

__all__ = ["apply_solve"]


class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, forward, backward):  # noqa: D102
        ctx.solve = backward
        ctx.real = not y.is_complex()
        ctx.dtype = y.dtype
        with torch.no_grad():
            return forward(y)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        g = ctx.solve(grad.resolve_conj().contiguous())
        return (g.real.to(ctx.dtype) if ctx.real else g.to(ctx.dtype)), None, None


def apply_solve(y, forward, backward) -> torch.Tensor:
    """``forward(y)``, recorded so that the backward pass is ``backward``.

    Parameters
    ----------
    y : torch.Tensor
        The data the solve is differentiated with respect to.
    forward : callable
        The solve itself, taking the data and answering the solution.
    backward : callable
        ``g -> A N^-1 g``: the adjoint of the solve, taking an incoming
        gradient of the solution and answering one of the data.
    """
    return _Solve.apply(y, forward, backward)
