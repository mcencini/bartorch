"""Proximal operators, in torch.

A proximal step written here is one a torch graph can hold, which is what an
unrolled network needs and what BART's own regularizers -- reachable through
:func:`bartorch.tools.pics`, and faster -- cannot be part of.

    from bartorch import alg, linop, prox

    A = linop.Sense(maps, image_shape, traj=traj)
    g = prox.L1(A.ishape, weight=0.01)
    x = alg.ProximalGradient(A, kspace, g, max_iter=60).run()
"""

from __future__ import annotations

from bartorch.prox.base import Prox
from bartorch.prox.basic import L1, L2Ball, L2Squared, Stack, Zero, soft_threshold

__all__ = ["L1", "L2Ball", "L2Squared", "Prox", "Stack", "Zero", "soft_threshold"]
