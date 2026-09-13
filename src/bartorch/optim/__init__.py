"""BART's iterative algorithms, one class per algorithm.

A solver is configured once and called as ``solver(y, A, x0=None)``.  The
iteration is BART's, the one ``pics`` or ``nlinv`` runs: for the proximal
solvers the loop is written out in :mod:`bartorch.optim.iterators` and each
step calls the library, and for the rest the whole solve is one call into it.
``solver.in_library(y, A)`` is BART's own loop, and answers with the same bits.
"""

from __future__ import annotations

from bartorch.optim.linear import (
    ADMM,
    CG,
    FISTA,
    IST,
    NIHT,
    PRIDU,
    EulerMaruyama,
    Tikhonov,
    maxeigen,
)
from bartorch.optim.nonlinear import IRGNM
from bartorch.optim.scaling import data_scaling

__all__ = [
    "ADMM",
    "CG",
    "EulerMaruyama",
    "FISTA",
    "IRGNM",
    "IST",
    "NIHT",
    "PRIDU",
    "Tikhonov",
    "data_scaling",
    "maxeigen",
]
