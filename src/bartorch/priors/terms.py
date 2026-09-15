"""BART's regularization terms, one class per ``pics -R`` letter.

Total generalized variation and the two infimal convolutions extend the
optimization variable, which BART counts across the whole set of terms, so
they cannot be built one at a time.  :func:`bartorch.tools.pics` takes them,
and so do :class:`bartorch.optim.ADMM` and :class:`bartorch.optim.PRIDU`,
which are the iterations BART gives a term's transform to; the solve is then
the library's own loop over the enlarged variable, so it does not unroll.
"""

from __future__ import annotations

from bartorch.priors.base import Regularizer

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


#: BART's `FFT_FLAGS`: the three spatial axes, which are the image's last
#: three in C order.
_FFT_FLAGS = 7


def _both_kinds_of_axis(term, ndim: int | None) -> None:
    """Refuse an infimal convolution that has no axis of each kind.

    ``ictv_reg`` and ``ictgv_reg`` (``iter/tgv.c``) each assert
    ``flags & FFT_FLAGS`` and ``flags & ~FFT_FLAGS``: the convolution splits
    the image into a part smooth over the spatial axes and a part smooth over
    the rest, so both must exist.  An assertion aborts the process, so the
    same question is asked here.  Plain total generalized variation asserts
    neither.
    """
    xflags, _ = term._flags(ndim)
    if 0 != (xflags & _FFT_FLAGS) and 0 != (xflags & ~_FFT_FLAGS):
        return
    kind = "spatial" if xflags & _FFT_FLAGS else "non-spatial"
    over = "" if ndim is None else f" of an image of {ndim} axes"
    raise ValueError(
        f"{term!r} needs at least one of the image's last three axes and at least one "
        f"axis before them: the infimal convolution separates what is smooth over the "
        f"one from what is smooth over the other, so it needs both to exist. "
        f"axes={term.axes}{over} gives only {kind} axes. BART states this as an "
        f"assertion in iter/tgv.c, which would end the process rather than raise"
    )


def _pair(values, name: str) -> tuple[float, float]:
    pair = tuple(float(v) for v in values)
    if len(pair) != 2:
        raise ValueError(f"{name} is a pair of weights, not {values!r}")
    return pair


class TotalGeneralizedVariation(_Weighted):
    """Total generalized variation over ``axes`` (``pics -R G``).

    Adds unknowns to the optimization: :func:`bartorch.tools.pics`,
    :class:`bartorch.optim.ADMM` and :class:`bartorch.optim.PRIDU` take it, and
    nothing else does.  See the module's introduction.

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

    Adds unknowns to the optimization: :func:`bartorch.tools.pics`,
    :class:`bartorch.optim.ADMM` and :class:`bartorch.optim.PRIDU` take it, and
    nothing else does.  See the module's introduction.

    The infimal convolution separates what is smooth over one set of axes
    from what is smooth over the other, so ``axes`` must name at least one of
    the image's last three -- BART's spatial axes -- and at least one before
    them, typically the coefficients of a subspace or the frames of a series.

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

    def _check(self, ndim: int | None) -> None:
        _both_kinds_of_axis(self, ndim)


class InfimalConvolutionTGV(_Weighted):
    """Infimal convolution of total generalized variation over ``axes`` (``pics -R V``).

    Adds unknowns to the optimization: :func:`bartorch.tools.pics`,
    :class:`bartorch.optim.ADMM` and :class:`bartorch.optim.PRIDU` take it, and
    nothing else does.  See the module's introduction.

    The infimal convolution separates what is smooth over one set of axes
    from what is smooth over the other, so ``axes`` must name at least one of
    the image's last three -- BART's spatial axes -- and at least one before
    them, typically the coefficients of a subspace or the frames of a series.

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

    def _check(self, ndim: int | None) -> None:
        _both_kinds_of_axis(self, ndim)
