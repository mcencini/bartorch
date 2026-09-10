"""Linear operators: BART's own, and ones written here that BART can drive.

Every operator is a class.  :class:`LinearOperator` is what they have in
common and what to write a solver, a model or an interoperability layer
against; :class:`BartLinearOperator` is one that a BART handle stands behind,
which all the concrete ones are.

Two operators compose into BART's own composite rather than into a Python
chain::

    A = Sense(maps, (8, 128, 128), traj=traj)
    x = A.lstsq(kspace, lambda_=0.01)

so the conjugate gradients that solves that never return to Python.  An
operator written here enters the same way, through :class:`Callback` or by
subclassing :class:`LinearOperator`, and chains with BART's own.

Applying one to a tensor that requires a gradient records it, and the backward
pass is the adjoint, so an operator drops into a torch model as it stands.
:meth:`LinearOperator.to_deepinv` hands one over as a ``LinearPhysics``.
"""

from __future__ import annotations

from bartorch.linop.base import (
    Add,
    Adjoint,
    BartLinearOperator,
    Compose,
    LinearOperator,
)
from bartorch.linop.basic import (
    FFT,
    Callback,
    Diagonal,
    MultiplySum,
    Sampling,
)
from bartorch.linop.nufft import NUFFT
from bartorch.linop.sense import Sense

__all__ = [
    "Add",
    "Adjoint",
    "BartLinearOperator",
    "Callback",
    "Compose",
    "Diagonal",
    "FFT",
    "LinearOperator",
    "MultiplySum",
    "NUFFT",
    "Sampling",
    "Sense",
]
