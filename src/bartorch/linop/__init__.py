"""Linear operators, BART's own and Python-defined, composable into one BART operator."""

from __future__ import annotations

from bartorch.linop.base import LinearOperator
from bartorch.linop.basic import (
    FFT,
    Callback,
    Conj,
    Diagonal,
    Identity,
    MultiplySum,
    Sampling,
    Zero,
)
from bartorch.linop.nufft import NUFFT
from bartorch.linop.sense import Sense
from bartorch.linop.shape import (
    Extract,
    Flip,
    Mean,
    Pad,
    Permute,
    Real,
    Repeat,
    Reshape,
    Resize,
    Roll,
    ScaledSum,
    Sum,
    Transpose,
)

__all__ = [
    "Callback",
    "Conj",
    "Diagonal",
    "Extract",
    "FFT",
    "Flip",
    "Identity",
    "LinearOperator",
    "Mean",
    "MultiplySum",
    "NUFFT",
    "Pad",
    "Permute",
    "Real",
    "Repeat",
    "Reshape",
    "Resize",
    "Roll",
    "ScaledSum",
    "Sampling",
    "Sense",
    "Sum",
    "Transpose",
    "Zero",
]
