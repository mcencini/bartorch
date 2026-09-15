"""BART's iterative algorithms: a block per step, and a solver and a function per algorithm.

A block is one step as a torch module, which a network stacks; a solver loops
a block to BART's schedule, called as ``solver(y, A, x0=None)``; the function
``optim.fista(y, A, term)`` is that call in one expression.
"""

from __future__ import annotations

from bartorch.optim.blocks import ADMMBlock, FISTABlock, ISTBlock, PRIDUBlock
from bartorch.optim.functional import (
    admm,
    cg,
    fista,
    irgnm,
    ist,
    niht,
    pridu,
)
from bartorch.optim.linear import (
    ADMM,
    CG,
    FISTA,
    IST,
    NIHT,
    PRIDU,
    Tikhonov,
    maxeigen,
)
from bartorch.optim.nonlinear import IRGNM
from bartorch.optim.scaling import data_scaling

__all__ = [
    "ADMM",
    "ADMMBlock",
    "CG",
    "FISTA",
    "FISTABlock",
    "IRGNM",
    "IST",
    "ISTBlock",
    "NIHT",
    "PRIDU",
    "PRIDUBlock",
    "Tikhonov",
    "admm",
    "cg",
    "data_scaling",
    "fista",
    "irgnm",
    "ist",
    "niht",
    "maxeigen",
    "pridu",
]
