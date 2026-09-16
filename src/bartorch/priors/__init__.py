"""BART's regularization terms, as objects a solver in :mod:`bartorch.optim` takes,
and BART's denoisers.

A term names a regularization functional and its parameters, with axes as
indices rather than bitmasks; BART builds the proximal operator, and the linear
transform in front of it, from that.
"""

from __future__ import annotations

from bartorch.priors import denoise
from bartorch.priors.base import Regularizer, frozen
from bartorch.priors.denoise import *  # noqa: F401,F403
from bartorch.priors.implicit import ImplicitPrior
from bartorch.priors.terms import (
    L1,
    L2,
    FourierL1,
    ImageNIHT,
    ImaginaryL1,
    ImaginaryL2,
    InfimalConvolutionTGV,
    InfimalConvolutionTV,
    Laplace,
    LocallyLowRank,
    NonNegative,
    TotalGeneralizedVariation,
    TotalVariation,
    Wavelet,
    WaveletNIHT,
)

__all__ = [
    "FourierL1",
    "ImageNIHT",
    "ImplicitPrior",
    "ImaginaryL1",
    "ImaginaryL2",
    "InfimalConvolutionTGV",
    "InfimalConvolutionTV",
    "L1",
    "L2",
    "Laplace",
    "LocallyLowRank",
    "NonNegative",
    "Regularizer",
    "TotalGeneralizedVariation",
    "TotalVariation",
    "Wavelet",
    "WaveletNIHT",
    "frozen",
    *denoise.__all__,
]
