"""The linear solve as a torch autograd function; the backward pass is another solve.

``x = N^-1 A^H y`` with ``N = A^H A + lambda I`` is linear in ``y``, so the
vector-Jacobian product is ``A N^-1``: one further solve with the same normal
operator, followed by one forward application.  The solution is recorded as a
single operation rather than by unrolling the iteration, as
``src/nlops/norm_inv.c`` does.

Three consequences follow.  A warm start receives no gradient, the solution of
a linear system being independent of the starting point.  The backward solve is
only as accurate as its own iteration count.  And no gradient reaches the
operator's own data -- sensitivities, a trajectory, a term's weight -- and no
second derivative is available, the backward solve running inside the library
and recording nothing itself.
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
