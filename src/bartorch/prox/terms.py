"""BART's regularization terms, one class per ``pics -R`` letter.

Total generalized variation and the two infimal convolutions are absent: they
extend the optimization variable, which BART counts across the whole set of
terms, so they cannot be built one at a time.  :func:`bartorch.tools.pics`
reaches them.
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

#: The wavelet families ``opt_reg_configure`` accepts.
_FAMILIES = ("haar", "dau2", "cdf44")


def _axes(axes) -> tuple[int, ...]:
    return (axes,) if isinstance(axes, int) else tuple(axes)


def _family(family: str) -> str:
    if family not in _FAMILIES:
        raise ValueError(f"wavelet family must be one of {_FAMILIES}, not {family!r}")
    return family


class _Weighted(Regularizer):
    """A term over ``axes``, acting jointly along ``joint_axes``, with a weight."""

    def __init__(self, axes, weight: float, joint_axes=()):
        self.axes = _axes(axes)
        self.joint_axes = _axes(joint_axes)
        self.weight = float(weight)


class Wavelet(_Weighted):
    """l1 norm of the wavelet transform over ``axes`` (``pics -R W``).

    Parameters
    ----------
    axes : int or tuple of int
        Axes to transform, as indices into the image's shape.
    weight : float
    joint_axes : int or tuple of int, optional
        Axes along which a coefficient is kept or zeroed together.
    family : {'haar', 'dau2', 'cdf44'}
        Wavelet family (``pics --wavelet``).
    randshift : bool
        Cycle-spin the transform by a random shift; ``pics -n`` turns it off.

    Examples
    --------
    >>> Wavelet(axes=(-1, -2), weight=0.005)
    """

    kind = "W"

    def __init__(
        self, axes, weight: float, joint_axes=(), *, family: str = "dau2", randshift: bool = True
    ):
        super().__init__(axes, weight, joint_axes)
        self.family = _family(family)
        self.randshift = bool(randshift)

    def _options(self) -> tuple[int, str, int]:
        return 8, self.family, int(self.randshift)


class TotalVariation(_Weighted):
    """l1 norm of the finite differences over ``axes`` (``pics -R T``)."""

    kind = "T"


class LocallyLowRank(_Weighted):
    """Nuclear norm of blocks over ``axes`` (``pics -R L``).

    Parameters
    ----------
    axes : int or tuple of int
        Axes the blocks span, as indices into the image's shape.
    weight : float
    joint_axes : int or tuple of int, optional
        Axes forming the columns of each block's matrix.
    block : int
        Block edge length (``pics -b``).
    randshift : bool
        Shift the block grid by a random offset; ``pics -n`` turns it off.
    overlapping : bool
        Fully overlapping blocks instead of a shifted grid (``pics -N``).
    """

    kind = "L"

    def __init__(
        self,
        axes,
        weight: float,
        joint_axes=(),
        *,
        block: int = 8,
        randshift: bool = True,
        overlapping: bool = False,
    ):
        super().__init__(axes, weight, joint_axes)
        self.block = int(block)
        self.randshift = bool(randshift)
        self.overlapping = bool(overlapping)

    def _options(self) -> tuple[int, str, int]:
        return self.block, "dau2", 2 if self.overlapping else int(self.randshift)


class Laplace(_Weighted):
    """Laplacian penalty over ``axes`` (``pics -R P``)."""

    kind = "P"


class FourierL1(_Weighted):
    """l1 norm of the Fourier transform over ``axes`` (``pics -R F``)."""

    kind = "F"


class _Joint(Regularizer):
    """A term with no axes of its own."""

    def __init__(self, weight: float, joint_axes=()):
        self.joint_axes = _axes(joint_axes)
        self.weight = float(weight)


class L1(_Joint):
    """l1 norm of the image (``pics -R I``)."""

    kind = "I"


class ImaginaryL1(_Joint):
    """l1 norm of the image's imaginary part (``pics -R R1``)."""

    kind = "R1"


class ImaginaryL2(_Joint):
    """Squared l2 norm of the image's imaginary part (``pics -R R2``)."""

    kind = "R2"


class L2(Regularizer):
    """Squared l2 norm of the image (``pics -R Q``), which is what ``pics -r`` adds."""

    kind = "Q"

    def __init__(self, weight: float):
        self.weight = float(weight)


class NonNegative(Regularizer):
    """Projection onto non-negative images (``pics -R S``); it has no weight."""

    kind = "S"


class _Counted(Regularizer):
    """A term that keeps the ``count`` largest entries instead of thresholding."""

    def __init__(self, axes, count: int, joint_axes=()):
        self.axes = _axes(axes)
        self.joint_axes = _axes(joint_axes)
        self.count = int(count)


class WaveletNIHT(_Counted):
    """Keep the ``count`` largest wavelet coefficients over ``axes`` (``pics -R H``).

    Parameters
    ----------
    axes : int or tuple of int
        Axes to transform, as indices into the image's shape.
    count : int
    joint_axes : int or tuple of int, optional
        Axes along which a coefficient is kept or zeroed together.
    family : {'haar', 'dau2', 'cdf44'}
        Wavelet family (``pics --wavelet``).
    randshift : bool
        Cycle-spin the transform by a random shift; ``pics -n`` turns it off.
    """

    kind = "H"

    def __init__(
        self, axes, count: int, joint_axes=(), *, family: str = "dau2", randshift: bool = True
    ):
        super().__init__(axes, count, joint_axes)
        self.family = _family(family)
        self.randshift = bool(randshift)

    def _options(self) -> tuple[int, str, int]:
        return 8, self.family, int(self.randshift)


class ImageNIHT(_Counted):
    """Keep the ``count`` largest image entries (``pics -R N``)."""

    kind = "N"
