"""Iterative algorithms, in torch.

Each is a step rather than a loop, so the loop belongs to the caller::

    from bartorch import alg, linop, prox

    A = linop.Sense(maps, (8, 128, 128), traj=traj)
    x = alg.ConjugateGradient(A, kspace, lambda_=0.01, max_iter=30).run()

Why these exist beside BART's own
---------------------------------
BART's solvers are already here twice over -- :func:`bartorch.tools.pics` and
:func:`bartorch.tools.nlinv` for the assembled problem,
:meth:`bartorch.linop.LinearOperator.lstsq` and
:meth:`bartorch.nlop.NonlinearOperator.irgnm` for an operator -- and they are
faster than anything in this module, because their loop never returns to
Python between steps.  Reach for one of those to reconstruct something.

What they cannot be is part of a torch graph.  These can: an iteration written
as a step is one an unrolled network drives itself, with a learned proximal
operator in place of :mod:`bartorch.prox`'s, and the gradient of the whole
unrolled loop comes back through the encoding because a
:class:`~bartorch.linop.LinearOperator` differentiates.
"""

from __future__ import annotations

from bartorch.alg.base import Algorithm
from bartorch.alg.linear import ConjugateGradient
from bartorch.alg.proximal import GradientDescent, ProximalGradient, max_eigenvalue_step

__all__ = [
    "Algorithm",
    "ConjugateGradient",
    "GradientDescent",
    "ProximalGradient",
    "max_eigenvalue_step",
]
