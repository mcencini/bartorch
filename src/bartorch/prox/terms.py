"""BART's regularization terms, one class per ``pics -R`` letter.

Total generalized variation and the two infimal convolutions extend the
optimization variable, which BART counts across the whole set of terms, so
they cannot be built one at a time: :func:`bartorch.tools.pics` takes them,
and a solver in :mod:`bartorch.optim` does not.
"""

from __future__ import annotations

from bartorch.prox.base import Regularizer

__all__ = [
    "FourierL1",
    "ImageNIHT",
    "ImaginaryL1",
    "ImaginaryL2",
    "InfimalConvolutionTGV",
    "InfimalConvolutionTV",
    "L1",
    "L2",
    "Laplace",
    "LocallyLowRank",
    "NonNegative",
    "TotalGeneralizedVariation",
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

    def _settings(self) -> dict[str, object]:
        return {"family": self.family, "randshift": self.randshift}


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

    def _settings(self) -> dict[str, object]:
        return {"block": self.block, "randshift": self.randshift, "overlapping": self.overlapping}


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

    def _settings(self) -> dict[str, object]:
        return {"family": self.family, "randshift": self.randshift}


class ImageNIHT(_Counted):
    """Keep the ``count`` largest image entries (``pics -R N``)."""

    kind = "N"


def _pair(values, name: str) -> tuple[float, float]:
    pair = tuple(float(v) for v in values)
    if len(pair) != 2:
        raise ValueError(f"{name} is a pair of weights, not {values!r}")
    return pair


class TotalGeneralizedVariation(_Weighted):
    """Total generalized variation over ``axes`` (``pics -R G``).

    Only :func:`bartorch.tools.pics` takes it; see the module's introduction.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to differentiate, as indices into the image's shape.
    weight : float
    joint_axes : int or tuple of int, optional
    alpha : tuple of float
        BART's ``alpha1:alpha0`` pair (``pics --alpha``).
    """

    kind = "G"
    _extends = True

    def __init__(self, axes, weight: float, joint_axes=(), *, alpha=(1.0, 3.0**0.5)):
        super().__init__(axes, weight, joint_axes)
        self.alpha = _pair(alpha, "alpha")

    def _settings(self) -> dict[str, object]:
        return {"alpha": self.alpha}


class InfimalConvolutionTV(_Weighted):
    """Infimal convolution of total variation over ``axes`` (``pics -R C``).

    Only :func:`bartorch.tools.pics` takes it; see the module's introduction.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to differentiate, as indices into the image's shape.
    weight : float
    joint_axes : int or tuple of int, optional
    gamma : tuple of float
        BART's ``gamma1:gamma2`` pair (``pics --gamma``).
    """

    kind = "C"
    _extends = True

    def __init__(self, axes, weight: float, joint_axes=(), *, gamma=(1.0, 1.0)):
        super().__init__(axes, weight, joint_axes)
        self.gamma = _pair(gamma, "gamma")

    def _settings(self) -> dict[str, object]:
        return {"gamma": self.gamma}


class InfimalConvolutionTGV(_Weighted):
    """Infimal convolution of total generalized variation over ``axes`` (``pics -R V``).

    Only :func:`bartorch.tools.pics` takes it; see the module's introduction.

    Parameters
    ----------
    axes : int or tuple of int
        Axes to differentiate, as indices into the image's shape.
    weight : float
    joint_axes : int or tuple of int, optional
    alpha : tuple of float
        BART's ``alpha1:alpha0`` pair (``pics --alpha``).
    gamma : tuple of float
        BART's ``gamma1:gamma2`` pair (``pics --gamma``).
    """

    kind = "V"
    _extends = True

    def __init__(
        self, axes, weight: float, joint_axes=(), *, alpha=(1.0, 3.0**0.5), gamma=(1.0, 1.0)
    ):
        super().__init__(axes, weight, joint_axes)
        self.alpha = _pair(alpha, "alpha")
        self.gamma = _pair(gamma, "gamma")

    def _settings(self) -> dict[str, object]:
        return {"alpha": self.alpha, "gamma": self.gamma}
