"""Nonlinear operators, BART's own and Python-defined, composable with linear ones.

An operator maps one or more inputs to one or more outputs, as BART's
``nlop_s`` does.  ``@`` composes, :meth:`~bartorch.nlop.NonlinearOperator.partial`
fixes an input, and :meth:`~bartorch.nlop.NonlinearOperator.linearize` gives
the derivative at a point.
"""

from __future__ import annotations

from bartorch.nlop.base import NonlinearOperator
from bartorch.nlop.basic import (
    Abs,
    Add,
    Constant,
    Divide,
    Exp,
    Inverse,
    Log,
    Multiply,
    Phase,
    Power,
    RootSumOfSquares,
    SmoothAbs,
    Sqrt,
    Sum,
    Weighted,
)
from bartorch.nlop.callback import TorchOperator
from bartorch.nlop.irgnm import IRGNM, IRGNMBlock, irgnm
from bartorch.nlop.mri import (
    CartesianSense,
    CoilSense,
    NoncartesianSense,
    NonlinearSense,
)
from bartorch.nlop.simulation import (
    Bloch,
    InversionRecovery,
    MultiEcho,
    SignalModel,
)

__all__ = [
    "Abs",
    "Add",
    "Bloch",
    "CartesianSense",
    "CoilSense",
    "Constant",
    "Divide",
    "Exp",
    "Inverse",
    "InversionRecovery",
    "IRGNM",
    "irgnm",
    "IRGNMBlock",
    "Log",
    "MultiEcho",
    "Multiply",
    "NoncartesianSense",
    "NonlinearOperator",
    "NonlinearSense",
    "Phase",
    "Power",
    "RootSumOfSquares",
    "SignalModel",
    "SmoothAbs",
    "Sqrt",
    "Sum",
    "TorchOperator",
    "Weighted",
]
