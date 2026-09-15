"""BART's iterative algorithms, as a class and a function per algorithm.

A solver is configured once and called as ``solver(y, A, x0=None)``; the
function ``optim.fista(y, A, term)`` is that call in one expression.  The
iteration is BART's, the one ``pics`` or ``nlinv`` runs: the proximal solvers
step through Python, so that a solver can be unrolled, and the rest solve in
one call into the library.  ``solver.in_library(y, A)`` is BART's own loop
either way, and answers with the same bits.
"""

from __future__ import annotations

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
    "CG",
    "FISTA",
    "IRGNM",
    "IST",
    "NIHT",
    "PRIDU",
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
