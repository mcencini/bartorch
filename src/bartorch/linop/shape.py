"""Operators that rearrange, reduce or restrict a shape, each one BART constructor.

BART counts its dimensions the other way round and always at
:data:`~bartorch._lib.DIMS` of them, so a C-order shape, an axis, a set of
axes and a per-axis vector each have to be turned around on the way in.  The
helpers at the top of this file do that in one place; no constructor below
does its own arithmetic on an index.
"""

from __future__ import annotations

import math

from bartorch import _marshal
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, axes_flags, dims
from bartorch.linop.base import LinearOperator

__all__ = [
    "Extract",
    "Flip",
    "Hankel",
    "Mean",
    "Pad",
    "Permute",
    "Real",
    "Repeat",
    "Reshape",
    "Resize",
    "Roll",
    "ScaledSum",
    "Sum",
    "Transpose",
]

#: What :class:`Pad` and :class:`Roll` fill the new entries with, under numpy's
#: names for the same things.  BART's own are PAD_VALID, PAD_SAME, PAD_CYCLIC,
#: PAD_SYMMETRIC, PAD_REFLECT and PAD_CAUSAL, in that order.
_PADDING = {"constant": 0, "wrap": 2, "symmetric": 3, "reflect": 4}


def _padding(mode: str) -> int:
    if mode not in _PADDING:
        raise ValueError(f"padding mode {mode!r} is not one of {sorted(_PADDING)}")
    return _PADDING[mode]


def _axis(axis: int, ndim: int) -> int:
    """A C-order axis index as BART's, counting from the other end."""
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of range for {ndim} axes")
    return ndim - 1 - (axis % ndim)


def _reduced(shape: Shape, axes) -> tuple[int, ...]:
    """``shape`` with the given axes set to one, which is what BART reduces to."""
    ndim = len(shape)
    keep = {a % ndim for a in (axes if isinstance(axes, (tuple, list)) else (axes,))}
    return tuple(1 if i in keep else n for i, n in enumerate(shape))


class Real(LinearOperator):
    """The real part, BART's ``linop_zreal``.

    Real only in value: the result is still complex, with zero imaginary part,
    because that is the one kind of array BART's operators pass between them.

    Like :class:`~bartorch.linop.Conj` this is linear over the reals and not
    over the complex numbers, so it is self-adjoint for a real inner product
    and fails a complex dot test on its own.  Composed between operators that
    are complex-linear it behaves like any other term; alone it is a
    projection, and that is what it is for.

    Parameters
    ----------
    shape : tuple of int
        The shape it maps to itself, C order.
    """

    def __init__(self, shape: Shape):
        self._shape = tuple(shape)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(library().bartorch_linop_zreal, DIMS, dims(self._shape))
        return Built(ptr, self._shape, self._shape)


class Sum(LinearOperator):
    """Sum over ``axes``, BART's ``linop_sum``.

    The summed axes are kept with size one rather than dropped, which is what
    BART does and what lets the result broadcast back against the input.  Its
    adjoint is :class:`Repeat`.

    BART attaches a closed-form pseudo-inverse to this operator, but it
    answers for :class:`ScaledSum`'s scaling, so
    :meth:`~bartorch.linop.LinearOperator.pinv` takes the solver here.

    Parameters
    ----------
    shape : tuple of int
        The shape it sums over, C order.
    axes : int or tuple of int
        Which axes to sum, as indices into ``shape``.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.ishape = self._shape
        self.oshape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_sum, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class ScaledSum(LinearOperator):
    """Sum over ``axes``, divided by the square root of how many were summed.

    BART's ``linop_scaled_sum``.  The scaling is what makes it well behaved:
    its normal operator is an orthogonal projection rather than a multiple of
    one, so its spectral norm is one and
    :meth:`~bartorch.linop.LinearOperator.pinv` has a closed form -- BART
    solves ``(A^H A + damp I) x = A^H y`` here directly, with no iteration.

    :class:`Sum` is the plain sum; this is the one to use inside a solver,
    where the conditioning and the exact inverse are worth the factor.

    Parameters
    ----------
    shape : tuple of int
        The shape it sums over, C order.
    axes : int or tuple of int
        Which axes to sum, as indices into ``shape``.
    """

    # Checked against an exact damped solve in tests/test_linop_shape.py, for
    # several sizes and dampings.  BART's routine is right for this operator
    # and wrong for the plain sum it is shared with, so this is where the flag
    # goes and nowhere else.
    _exact_pinv = True

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.ishape = self._shape
        self.oshape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_scaled_sum, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class Mean(LinearOperator):
    """Average over ``axes``, BART's ``linop_avg``.

    :class:`Sum` divided by how many were summed, and like it the axes are kept
    with size one.

    Parameters
    ----------
    shape : tuple of int
        The shape it averages over, C order.
    axes : int or tuple of int
        Which axes to average, as indices into ``shape``.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.ishape = self._shape
        self.oshape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_avg, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class Repeat(LinearOperator):
    """Repeat along ``axes`` up to ``shape``, BART's ``linop_repmat``.

    The adjoint of :class:`Sum`: the domain is ``shape`` with the repeated axes
    set to one, so this is the broadcast that ``Sum`` undoes.

    Parameters
    ----------
    shape : tuple of int
        The codomain, C order -- what the repeating produces.
    axes : int or tuple of int
        Which axes to repeat along.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self.oshape = self._shape
        self.ishape = _reduced(self._shape, axes)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_repmat, DIMS, dims(self._shape), flags)
        return Built(ptr, self.ishape, self.oshape)


class Flip(LinearOperator):
    """Reverse ``axes``, BART's ``linop_flip``.

    Parameters
    ----------
    shape : tuple of int
        The shape it maps to itself, C order.
    axes : int or tuple of int
        Which axes to reverse.
    """

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self._shape))
        ptr = self._under_lock(library().bartorch_linop_flip, DIMS, dims(self._shape), flags)
        return Built(ptr, self._shape, self._shape)


class _Hankelization(LinearOperator):
    """``linop_hankelization``: the window on the first BART dimension going spare."""

    def __init__(self, ishape, oshape, axis: int, window: int):
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        self.axis, self.window = axis, window
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        ptr = self._under_lock(
            library().bartorch_linop_hankel,
            DIMS,
            dims(self.ishape),
            _axis(self.axis, ndim),
            ndim,
            self.window,
        )
        return Built(ptr, self.ishape, self.oshape)


def Hankel(shape: Shape, axis: int, window: int) -> LinearOperator:  # noqa: N802
    """A sliding window along ``axis``, BART's ``linop_hankelization``.

    ``torch.Tensor.unfold(axis, window, 1)`` as an operator, and the same
    thing as the trajectory matrix that singular spectrum analysis is built
    on: an axis of ``n`` becomes ``n - window + 1`` positions, each carrying
    the ``window`` samples that start there.  The codomain is the domain with
    that axis shortened and the window added as a last axis, which is where
    ``unfold`` puts it.

    BART makes the windows by striding rather than by copying, so the overlap
    costs nothing to build; the adjoint adds each sample back into every
    window it appeared in, which is what makes this an operator rather than a
    view.

    Parameters
    ----------
    shape : tuple of int
        The domain, C order.
    axis : int
        Which axis to slide along.
    window : int
        How many samples each position carries.  At most the length of the
        axis; equal to it gives one position.

    Examples
    --------
    >>> H = Hankel((64, 8), axis=0, window=16)
    >>> H.oshape
    (49, 8, 16)
    """
    from bartorch.linop.shape import Permute  # noqa: PLC0415  (itself, after definition)

    ishape = tuple(shape)
    ndim = len(ishape)
    axis %= ndim
    window = int(window)

    if window < 1:
        raise ValueError(f"a window of {window} has nothing in it")
    if window > ishape[axis]:
        raise ValueError(
            f"a window of {window} does not fit in axis {axis}, which is {ishape[axis]} long"
        )
    if ndim >= DIMS:
        raise ValueError(f"the window axis would take this past BART's {DIMS} dimensions")

    # BART puts the window on a dimension of its own, and the first one going
    # spare reads back as a leading axis in C order.  torch.unfold puts it
    # last, so one permute moves it there.
    positions = list(ishape)
    positions[axis] = ishape[axis] - window + 1
    leading = (window, *positions)

    out: LinearOperator = _Hankelization(ishape, leading, axis, window)
    order = (*range(1, ndim + 1), 0)
    return Permute(leading, order) @ out


class Reshape(LinearOperator):
    """Read the same elements under another shape, BART's ``linop_reshape``.

    The two shapes must hold the same number of elements.  Nothing moves: this
    is the identity on the buffer and a relabelling of its axes.

    Parameters
    ----------
    oshape, ishape : tuple of int
        Codomain and domain, C order.
    """

    def __init__(self, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        if math.prod(self.oshape) != math.prod(self.ishape):
            raise ValueError(
                f"cannot reshape {self.ishape} to {self.oshape}: "
                f"{math.prod(self.ishape)} elements against {math.prod(self.oshape)}"
            )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_reshape, DIMS, dims(self.oshape), DIMS, dims(self.ishape)
        )
        return Built(ptr, self.ishape, self.oshape)


class Resize(LinearOperator):
    """Crop or zero-fill about the centre of each axis, BART's ``resize -c``.

    An axis the codomain is shorter along is cropped, one it is longer along is
    filled with zeros, and either happens about the middle rather than the
    corner -- which is what a Fourier transform's conventions want.

    Parameters
    ----------
    oshape, ishape : tuple of int
        Codomain and domain, C order, with the same number of axes.
    """

    def __init__(self, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        if len(self.oshape) != len(self.ishape):
            raise ValueError(
                f"resizing keeps the axes it has: {self.ishape} and {self.oshape} differ in rank"
            )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_resize, DIMS, dims(self.oshape), dims(self.ishape)
        )
        return Built(ptr, self.ishape, self.oshape)


class Extract(LinearOperator):
    """Take the block of ``oshape`` that starts at ``start``, BART's ``linop_extract``.

    The restriction operator: its adjoint puts the block back where it came
    from and leaves the rest zero.  This is what indexing an operator with
    ``A[...]`` builds.

    Parameters
    ----------
    start : sequence of int
        Where the block begins, one offset per axis of ``ishape``.
    oshape, ishape : tuple of int
        The block and the shape it comes out of, C order.
    """

    def __init__(self, start, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        self.start = tuple(int(s) for s in start)
        if len(self.oshape) != len(self.ishape):
            raise ValueError(
                f"extracting keeps the axes it has: {self.ishape} and {self.oshape} differ in rank"
            )
        for axis, (begin, width, full) in enumerate(zip(self.start, self.oshape, self.ishape)):
            if begin < 0 or begin + width > full:
                raise ValueError(
                    f"axis {axis}: a block of {width} from {begin} does not fit in {full}"
                )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_extract,
            DIMS,
            _marshal.padded_offsets(self.start, len(self.ishape), "start"),
            dims(self.oshape),
            dims(self.ishape),
        )
        return Built(ptr, self.ishape, self.oshape)


class Transpose(LinearOperator):
    """Swap two axes, BART's ``linop_transpose``.

    :class:`Permute` is the general rearrangement; this is the pair swap that
    ``torch.transpose`` is.

    Parameters
    ----------
    shape : tuple of int
        The domain, C order.
    axis0, axis1 : int
        The two axes to exchange.
    """

    def __init__(self, shape: Shape, axis0: int, axis1: int):
        self._shape = tuple(shape)
        ndim = len(self._shape)
        self.axis0, self.axis1 = axis0 % ndim, axis1 % ndim
        if self.axis0 == self.axis1:
            raise ValueError(f"axis {axis0} cannot be exchanged with itself")
        self.ishape = self._shape
        swapped = list(self._shape)
        swapped[self.axis0], swapped[self.axis1] = swapped[self.axis1], swapped[self.axis0]
        self.oshape = tuple(swapped)
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        ptr = self._under_lock(
            library().bartorch_linop_transpose,
            DIMS,
            _axis(self.axis0, ndim),
            _axis(self.axis1, ndim),
            dims(self.ishape),
        )
        return Built(ptr, self.ishape, self.oshape)


class Permute(LinearOperator):
    """Rearrange the axes, BART's ``linop_permute``.

    ``order`` reads the way ``torch.permute``'s does: axis ``i`` of the result
    is axis ``order[i]`` of the input.

    Parameters
    ----------
    shape : tuple of int
        The domain, C order.
    order : sequence of int
        A permutation of ``range(len(shape))``.
    """

    def __init__(self, shape: Shape, order):
        self.ishape = tuple(shape)
        ndim = len(self.ishape)
        self.order = tuple(int(o) % ndim for o in order)
        if sorted(self.order) != list(range(ndim)):
            raise ValueError(f"{tuple(order)} is not a permutation of {ndim} axes")
        self.oshape = tuple(self.ishape[o] for o in self.order)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_permute,
            DIMS,
            _marshal.padded_order(self.order, len(self.ishape)),
            dims(self.ishape),
        )
        return Built(ptr, self.ishape, self.oshape)


class Roll(LinearOperator):
    """Shift along one axis, BART's ``linop_shift``.

    With ``mode="wrap"`` this is ``torch.roll``: what leaves one end comes back
    at the other.  With another mode what leaves is dropped and what arrives is
    whatever that mode supplies, so the operator is no longer unitary.

    Parameters
    ----------
    shape : tuple of int
        The shape it maps to itself, C order.
    shift : int
        How far to move, towards higher indices when positive.
    axis : int
        Which axis to move along.
    mode : str
        ``"wrap"``, ``"constant"``, ``"symmetric"`` or ``"reflect"``.
    """

    def __init__(self, shape: Shape, shift: int, axis: int = -1, mode: str = "wrap"):
        self._shape = tuple(shape)
        self.shift = int(shift)
        self.axis = axis % len(self._shape)
        self.mode = mode
        self._pad = _padding(mode)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_shift,
            DIMS,
            dims(self._shape),
            _axis(self.axis, len(self._shape)),
            self.shift,
            self._pad,
        )
        return Built(ptr, self._shape, self._shape)


class Pad(LinearOperator):
    """Make each axis longer, BART's ``linop_padding``.

    Parameters
    ----------
    shape : tuple of int
        The domain, C order.
    before, after : sequence of int
        How much to add at each end of each axis.  A single number applies to
        every axis.
    mode : str
        ``"constant"`` for zeros, or ``"wrap"``, ``"symmetric"`` or
        ``"reflect"`` to carry values in from the array itself.
    """

    def __init__(self, shape: Shape, before, after=None, mode: str = "constant"):
        self.ishape = tuple(shape)
        ndim = len(self.ishape)
        self.before = (before,) * ndim if isinstance(before, int) else tuple(before)
        after = self.before if after is None else after
        self.after = (after,) * ndim if isinstance(after, int) else tuple(after)
        if len(self.before) != ndim or len(self.after) != ndim:
            raise ValueError(f"before and after need one entry per axis, {ndim} of them")
        if any(b < 0 for b in self.before) or any(a < 0 for a in self.after):
            raise ValueError("padding cannot be negative; crop with Resize or Extract")
        self.mode = mode
        self._pad = _padding(mode)
        self.oshape = tuple(n + b + a for n, b, a in zip(self.ishape, self.before, self.after))
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        ptr = self._under_lock(
            library().bartorch_linop_padding,
            DIMS,
            dims(self.ishape),
            self._pad,
            _marshal.padded_offsets(self.before, ndim, "before"),
            _marshal.padded_offsets(self.after, ndim, "after"),
        )
        return Built(ptr, self.ishape, self.oshape)
