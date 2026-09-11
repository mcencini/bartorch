"""Nonlinear operators: BART's own, and ones written here that BART can drive.

BART's nonlinear operator carries a derivative and the adjoint of that
derivative at the point it was last evaluated, which is what a forward- and a
reverse-mode product are.  So a torch model goes to BART's Gauss-Newton
solver::

    F = FromTorch(signal_model, ishape=(2,), oshape=t.shape)
    p = F.irgnm(measured, x0=guess)

and a BART operator differentiates in torch, and the two chain with each other
and with the linear operators in :mod:`bartorch.linop`.
"""

from __future__ import annotations

from bartorch.nlop.base import (
    BartNonlinearOperator,
    Chain,
    FromLinear,
    NonlinearOperator,
)
from bartorch.nlop.callback import Callback, FromTorch

__all__ = [
    "BartNonlinearOperator",
    "Callback",
    "Chain",
    "FromLinear",
    "FromTorch",
    "NonlinearOperator",
]
