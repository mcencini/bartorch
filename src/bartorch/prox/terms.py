"""BART's regularization terms, one class each.

Each is the term ``pics -R`` names with the same letter, and carries what that
term's specification carries and nothing more.  What the term *is* -- the
proximal operator and the transform beside it -- BART builds.

Three of BART's terms are not here: total generalized variation and the two
infimal convolutions extend the optimisation variable, and what they add is
counted across the whole set, so they cannot be built one at a time.
``bartorch.tools.pics(..., regularizers="G:3:0:...")`` reaches them.
"""

from __future__ import annotations

from bartorch.prox.base import Regularizer

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
    "TotalVariation",
    "Wavelet",
    "WaveletNIHT",
]


class _Weighted(Regularizer):
    """A term over some axes, joined along others, with a weight."""

    def __init__(self, axes, weight: float, joint_axes=()):
        self.axes = tuple(axes) if not isinstance(axes, int) else (axes,)
        self.joint_axes = tuple(joint_axes) if not isinstance(joint_axes, int) else (joint_axes,)
        self.weight = float(weight)


class Wavelet(_Weighted):
    """l1 of the wavelet transform over ``axes`` (``pics -R W``).

    The usual compressed-sensing penalty: sparse in a wavelet basis.

    Parameters
    ----------
    axes : int or tuple of int
        The axes to transform, as indices into the image's shape.
    weight : float
        The penalty's weight.
    joint_axes : int or tuple of int, optional
        Axes the threshold is joined along, so that an entry is kept or
        dropped for all of them together.

    Examples
    --------
    >>> Wavelet(axes=(-1, -2), weight=0.005)
    """

    kind = "W"


class TotalVariation(_Weighted):
    """l1 of the finite difference over ``axes`` (``pics -R T``)."""

    kind = "T"


class LocallyLowRank(_Weighted):
    """Nuclear norm over blocks of ``axes`` (``pics -R L``).

    The block size is the solve's ``llr_block``, which is ``pics -b``.
    """

    kind = "L"


class Laplace(_Weighted):
    """A Laplacian penalty over ``axes`` (``pics -R P``)."""

    kind = "P"


class FourierL1(_Weighted):
    """l1 of the Fourier transform over ``axes`` (``pics -R F``)."""

    kind = "F"


class _Joint(Regularizer):
    """A term with no axes of its own, only a weight and what it joins."""

    def __init__(self, weight: float, joint_axes=()):
        self.joint_axes = tuple(joint_axes) if not isinstance(joint_axes, int) else (joint_axes,)
        self.weight = float(weight)


class L1(_Joint):
    """l1 of the image itself (``pics -R I``)."""

    kind = "I"


class ImaginaryL1(_Joint):
    """l1 of the image's imaginary part (``pics -R R1``)."""

    kind = "R1"


class ImaginaryL2(_Joint):
    """l2 of the image's imaginary part (``pics -R R2``)."""

    kind = "R2"


class L2(Regularizer):
    """Squared l2 of the image (``pics -R Q``).

    Tikhonov as a regularization term.  ``lambda_`` on the solve is the other
    way to ask for one, and is BART's ``-r``.
    """

    kind = "Q"

    def __init__(self, weight: float):
        self.weight = float(weight)


class NonNegative(Regularizer):
    """The constraint that the image is positive (``pics -R S``).

    A projection, so it carries no weight.
    """

    kind = "S"


class _Counted(Regularizer):
    """A term that keeps a fixed number of entries rather than thresholding."""

    def __init__(self, axes, count: int, joint_axes=()):
        self.axes = tuple(axes) if not isinstance(axes, int) else (axes,)
        self.joint_axes = tuple(joint_axes) if not isinstance(joint_axes, int) else (joint_axes,)
        self.count = int(count)


class WaveletNIHT(_Counted):
    """Keep the ``count`` largest wavelet coefficients (``pics -R H``)."""

    kind = "H"


class ImageNIHT(_Counted):
    """Keep the ``count`` largest image entries (``pics -R N``)."""

    kind = "N"
