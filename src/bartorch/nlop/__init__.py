"""Nonlinear operators, BART's own and Python-defined, composable with linear ones.

An operator maps many inputs to many outputs, as BART's ``nlop_s`` does, and
the algebra here is ``nlops/chain.h``: :func:`combine` puts two side by side,
:func:`chain` feeds one output into one input, and
:meth:`~bartorch.nlop.NonlinearOperator.link`,
:meth:`~bartorch.nlop.NonlinearOperator.dup`,
:meth:`~bartorch.nlop.NonlinearOperator.stack_inputs` and the permutations
rearrange what is left.  Arguments are counted BART's way -- outputs first,
then inputs.
"""

from __future__ import annotations

from bartorch.nlop.base import (
    Chain,
    FromLinear,
    NonlinearOperator,
    chain,
    combine,
)
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
from bartorch.nlop.bundle import Bundle
from bartorch.nlop.callback import FromTorch, Parameters
from bartorch.nlop.derivative import Derivative
from bartorch.nlop.irgnm import IRGNM, irgnm
from bartorch.nlop.mri import (
    CartesianSense,
    CoilSense,
    NoncartesianSense,
    NonlinearSense,
)
from bartorch.nlop.simulation import (
    Bloch,
    FromTorchSim,
    InversionRecovery,
    MultiEcho,
)

__all__ = [
    "Abs",
    "Add",
    "Bloch",
    "Bundle",
    "CartesianSense",
    "Chain",
    "CoilSense",
    "Constant",
    "Derivative",
    "Divide",
    "Exp",
    "FromLinear",
    "FromTorch",
    "FromTorchSim",
    "IRGNM",
    "InversionRecovery",
    "Inverse",
    "Log",
    "MultiEcho",
    "Multiply",
    "NoncartesianSense",
    "NonlinearOperator",
    "NonlinearSense",
    "Parameters",
    "Phase",
    "Power",
    "RootSumOfSquares",
    "SmoothAbs",
    "Sqrt",
    "Sum",
    "Weighted",
    "chain",
    "combine",
    "irgnm",
]
