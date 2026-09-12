"""Operators that filter, differentiate, decompose or multiply by a matrix.

Each is one of BART's constructors.  What they have in common with everything
else in :mod:`bartorch.linop` is that they map a tensor to a tensor, so each is
a class and each composes with ``@``, ``+`` and the rest of the algebra into a
single BART operator.
"""

from __future__ import annotations

import torch

from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, axes_flags, dims
from bartorch.linop.base import LinearOperator

__all__ = ["Convolve", "Gradient", "Matrix"]

#: numpy's names for what BART calls CONV_CYCLIC, TRUNCATED, VALID and EXTENDED.
_CONV_TYPE = {"wrap": 0, "same": 1, "valid": 2, "full": 3}

#: BART's conv_mode.
_CONV_MODE = {"symmetric": 0, "causal": 1, "anticausal": 2}


class Matrix(LinearOperator):
    """Multiplication by a matrix along one axis, BART's ``linop_matrix``.

    The matrix is an array like any other: it contracts the axis the domain
    and the codomain differ along and broadcasts over the axes it has only one
    of, so one operator can carry a different matrix per slice.

    Parameters
    ----------
    matrix : tensor
        The matrix, with the domain's size along the axis it contracts and the
        codomain's along the axis it produces.
    oshape, ishape : tuple of int
        Codomain and domain, C order, with the same number of axes.

    Examples
    --------
    A basis of ``k`` functions over ``t`` samples, applied along the first
    axis of ``(t, y, x)``:

    >>> B = Matrix(basis, (k, y, x), (t, y, x))
    """

    def __init__(self, matrix: torch.Tensor, oshape: Shape, ishape: Shape):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        if len(self.oshape) != len(self.ishape):
            raise ValueError(
                f"a matrix keeps the axes it has: {self.ishape} and {self.oshape} differ in rank"
            )
        self.matrix = as_operand(matrix, tuple(matrix.shape), "matrix")
        if len(self.matrix.shape) != len(self.ishape):
            raise ValueError(
                f"the matrix needs one axis per axis of the operator: "
                f"{tuple(self.matrix.shape)} against {self.ishape}"
            )
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_matrix,
            DIMS,
            dims(self.oshape),
            dims(self.ishape),
            dims(tuple(self.matrix.shape)),
            self.matrix.data_ptr(),
            device=self.matrix.device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.matrix,))


class Convolve(LinearOperator):
    """Convolution with a fixed kernel, BART's ``linop_conv``.

    Parameters
    ----------
    kernel : tensor
        What to convolve with, with one axis per axis of ``shape``; an axis it
        has only one of is left alone.
    shape : tuple of int
        The domain, C order.
    axes : int or tuple of int
        Which axes to convolve along.
    mode : str
        How the ends are treated, under numpy's names: ``"wrap"`` (circular),
        ``"same"`` (the input's size, truncated), ``"valid"`` (only where the
        kernel fits) or ``"full"`` (extended).
    direction : str, optional
        ``"symmetric"``, ``"causal"`` or ``"anticausal"``.  BART only plans a
        ``"valid"`` or ``"full"`` convolution as a causal one, so leaving this
        out picks causal for those two and symmetric for the others rather
        than failing inside BART's planner.
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        shape: Shape,
        axes,
        mode: str = "same",
        direction: str | None = None,
    ):
        if mode not in _CONV_TYPE:
            raise ValueError(f"mode {mode!r} is not one of {sorted(_CONV_TYPE)}")
        if direction is None:
            direction = "causal" if mode in ("valid", "full") else "symmetric"
        if direction not in _CONV_MODE:
            raise ValueError(f"direction {direction!r} is not one of {sorted(_CONV_MODE)}")
        if mode in ("valid", "full") and "causal" != direction:
            raise ValueError(
                f"BART plans a {mode!r} convolution only as a causal one, not {direction!r}"
            )

        self.ishape = tuple(shape)
        self.kernel = as_operand(kernel, tuple(kernel.shape), "kernel")
        self.axes, self.mode, self.direction = axes, mode, direction

        if len(self.kernel.shape) != len(self.ishape):
            raise ValueError(
                f"the kernel needs one axis per axis of the operator: "
                f"{tuple(self.kernel.shape)} against {self.ishape}"
            )

        ndim = len(self.ishape)
        convolved = {a % ndim for a in (axes if isinstance(axes, (tuple, list)) else (axes,))}
        widths = tuple(self.kernel.shape)
        self.oshape = tuple(
            _conv_size(mode, n, widths[axis]) if axis in convolved else n
            for axis, n in enumerate(self.ishape)
        )
        if any(n < 1 for n in self.oshape):
            raise ValueError(f"a {widths} kernel leaves nothing of {self.ishape} in {mode!r} mode")
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_conv,
            DIMS,
            axes_flags(self.axes, len(self.ishape)),
            _CONV_TYPE[self.mode],
            _CONV_MODE[self.direction],
            dims(self.oshape),
            dims(self.ishape),
            dims(tuple(self.kernel.shape)),
            self.kernel.data_ptr(),
            device=self.kernel.device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.kernel,))


def _conv_size(mode: str, n: int, width: int) -> int:
    if mode in ("wrap", "same"):
        return n
    return n - width + 1 if "valid" == mode else n + width - 1


class _Differences(LinearOperator):
    """``linop_grad``: the differences stacked in BART's own order."""

    def __init__(self, ishape, oshape, axes):
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        self.axes = axes
        super().__init__()

    def _create(self) -> Built:
        ndim = len(self.ishape)
        # The differences go on the first BART dimension the shape does not
        # use, which read back in C order is a new leading axis.
        ptr = self._under_lock(
            library().bartorch_linop_grad,
            DIMS,
            dims(self.ishape),
            ndim,
            axes_flags(self.axes, ndim),
        )
        return Built(ptr, self.ishape, self.oshape)


def Gradient(shape: Shape, axes) -> LinearOperator:  # noqa: N802  (it is a constructor)
    """Finite differences along ``axes``, stacked on a new leading axis.

    BART's ``linop_grad``.  The codomain is ``(len(axes), *shape)``: one
    difference per axis, side by side, which is what a total-variation term is
    built on.

    The difference is the forward one and the boundary is circular: component
    ``i`` at the last index along its axis is the first entry minus the last,
    not zero.  So the operator has a null space -- a constant maps to zero --
    which is the usual thing for a gradient and worth knowing before solving
    with one.

    **The components come out in ascending axis order**, whatever order
    ``axes`` names them in.  BART stacks them by bit position, and its bits
    run the opposite way to C-order axes, so this reverses that with BART's
    own ``linop_flip`` rather than leaving the caller to discover that
    ``axes=(0, 1)`` puts axis 1 first.

    Parameters
    ----------
    shape : tuple of int
        The domain, C order.
    axes : int or tuple of int
        Which axes to difference along.  Order does not matter: BART takes a
        set of axes, and the result is ordered by axis either way.
    """
    from bartorch.linop.shape import Flip

    ishape = tuple(shape)
    ndim = len(ishape)
    chosen = tuple(a % ndim for a in (axes if isinstance(axes, (tuple, list)) else (axes,)))
    if len(set(chosen)) != len(chosen):
        raise ValueError(
            f"{tuple(axes) if not isinstance(axes, int) else axes} names an axis twice"
        )
    if ndim >= DIMS:
        raise ValueError(f"the new axis would take this past BART's {DIMS} dimensions")

    oshape = (len(chosen), *ishape)
    out: LinearOperator = _Differences(ishape, oshape, axes)
    if len(chosen) > 1:
        out = Flip(oshape, axes=0) @ out
    return out
