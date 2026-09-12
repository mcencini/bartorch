"""Linear operators, BART's own and Python-defined, composable into one BART operator."""

from __future__ import annotations

from bartorch.linop.base import LinearOperator
from bartorch.linop.basic import (
    FFT,
    Callback,
    ComponentDiagonal,
    Conj,
    Diagonal,
    Identity,
    MultiplySum,
    Sampling,
    Zero,
)
from bartorch.linop.combine import block, block_diag, concatenate, hstack, stack
from bartorch.linop.nufft import NUFFT
from bartorch.linop.sense import Sense
from bartorch.linop.shape import (
    Extract,
    Flip,
    Hankel,
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
from bartorch.linop.signal import Convolve, Gradient, Matrix

__all__ = [
    "block",
    "block_diag",
    "concatenate",
    "hstack",
    "stack",
    "Callback",
    "ComponentDiagonal",
    "Conj",
    "Convolve",
    "Diagonal",
    "Extract",
    "FFT",
    "Flip",
    "Gradient",
    "Hankel",
    "Identity",
    "LinearOperator",
    "Matrix",
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
