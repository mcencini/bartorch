"""BART's regularization terms, as objects.

A term is named rather than spelled::

    from bartorch import alg, linop, prox

    A = linop.Sense(maps, (8, 128, 128), traj=traj)
    x = alg.solve(A, kspace, regularizers=prox.Wavelet(axes=(-1, -2), weight=0.005))

Nothing here computes a proximal step.  BART builds the proximal operator and
the transform beside it -- the wavelet transform under a wavelet threshold,
the gradient under total variation -- in ``opt_reg_configure``, and an object
is what that function reads: which term, over which axes, with what weight.
Filling BART's table from an object rather than from a ``-R`` string is the
only difference between this and the tool, which is why an axis can be an
axis here and a bitmask there.
"""

from __future__ import annotations

from bartorch.prox.base import Regularizer
from bartorch.prox.terms import (
    L1,
    L2,
    FourierL1,
    ImageNIHT,
    ImaginaryL1,
    ImaginaryL2,
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
    "ImaginaryL1",
    "ImaginaryL2",
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
]
