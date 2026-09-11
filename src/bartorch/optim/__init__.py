"""BART's iterative algorithms, one class per algorithm.

A solver is configured once and called as ``solver(y, A, x0=None)``.  The
iteration runs inside the library, as the one ``pics`` or ``nlinv`` runs.
"""

from __future__ import annotations

from bartorch.optim.linear import ADMM, CG, FISTA, IST, NIHT, PRIDU, EulerMaruyama
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
    "data_scaling",
]
