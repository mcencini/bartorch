"""The basic nonlinear operators, each one BART constructor.

:class:`Multiply` is the two-input product every model is built out of,
broadcast the way BART broadcasts.  Everything here works on ``complex
float``, as BART's nonlinear operators do: :class:`Abs` returns a real
quantity in a complex tensor with a zero imaginary part, as ``zabs`` does.
"""

from __future__ import annotations

import torch

from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, axes_flags, dims
from bartorch.nlop.base import NonlinearOperator, _built

__all__ = [
    "Abs",
    "Add",
    "Constant",
    "Divide",
    "Exp",
    "Inverse",
    "Log",
    "Multiply",
    "Phase",
    "Power",
    "RootSumOfSquares",
    "SmoothAbs",
    "Sqrt",
    "Sum",
    "Weighted",
]


def _broadcast(a: Shape, b: Shape) -> Shape:
    """The shape BART's tensor product gives two operands, which is torch's rule."""
    if len(a) != len(b):
        raise ValueError(f"{a} and {b} do not have the same rank; BART broadcasts axis by axis")
    out = []
    for axis, (m, n) in enumerate(zip(a, b)):
        if m != n and 1 != m and 1 != n:
            raise ValueError(f"axis {axis} is {m} and {n}, which do not broadcast")
        out.append(max(m, n))
    return tuple(out)


class _Elementwise(NonlinearOperator):
    """One input to one output of the same shape, through one BART constructor."""

    #: The library function, and whatever it takes after the dimensions.
    _fn: str = ""

    def __init__(self, shape: Shape):
        self._shape = tuple(shape)
        super().__init__()

    def _extra(self) -> tuple:
        return ()

    def _create(self) -> Built:
        ptr = self._under_lock(
            getattr(library(), self._fn), DIMS, dims(self._shape), *self._extra()
        )
        return Built(ptr, self._shape, self._shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._shape})"


class Exp(_Elementwise):
    """``exp(x)``, elementwise.  BART's ``zexp``."""

    _fn = "bartorch_nlop_zexp"

    def _bundle(self):
        from bartorch.nlop.bundle import diagonal

        return diagonal(self, Exp(self._shape))


class Log(_Elementwise):
    """``log(x)``, elementwise.  BART's ``zlog``."""

    _fn = "bartorch_nlop_zlog"

    def _bundle(self):
        from bartorch.nlop.bundle import diagonal

        return diagonal(self, Inverse(self._shape))


class Sqrt(_Elementwise):
    """``sqrt(x)``, elementwise.  BART's ``zsqrt``."""

    _fn = "bartorch_nlop_zsqrt"

    def _bundle(self):
        from bartorch.nlop.base import chain
        from bartorch.nlop.bundle import diagonal, scaled

        # `zsqrt_apply` fills the diagonal with 0.5 and divides it by the
        # value, so it is half the inverse of the root and not of the point.
        half = chain(chain(Sqrt(self._shape), Inverse(self._shape)), scaled(self._shape, 0.5))
        return diagonal(self, half)


class Abs(_Elementwise):
    """``|x|``, elementwise, as a complex tensor.  BART's ``zabs``.

    Not differentiable at zero; :class:`SmoothAbs` is the variant that is.
    """

    _fn = "bartorch_nlop_zabs"

    def _composition(self):
        """``nlop_zabs_create`` is ``zrss`` over no axes (``someops.c:626``)."""
        return RootSumOfSquares(self._shape, ())

    def _bundle(self):
        from bartorch.nlop.bundle import of_composition

        return of_composition(self, self._composition())


class SmoothAbs(_Elementwise):
    """``sqrt(|x|^2 + eps)``, elementwise.  BART's ``smo_abs``.

    Parameters
    ----------
    shape : tuple of int
        The shape it maps, C order.
    eps : float
        What keeps the derivative finite at zero.
    """

    _fn = "bartorch_nlop_smo_abs"

    def __init__(self, shape: Shape, eps: float = 1e-12):
        self.eps = float(eps)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.eps,)

    def _composition(self):
        """``nlop_smo_abs_create`` is ``zrss`` over no axes, regularised (``someops.c:621``)."""
        return RootSumOfSquares(self._shape, (), self.eps)

    def _bundle(self):
        from bartorch.nlop.bundle import of_composition

        return of_composition(self, self._composition())

    def __repr__(self) -> str:
        return f"SmoothAbs({self._shape}, eps={self.eps})"


class Inverse(_Elementwise):
    """``1 / x``, elementwise.  BART's ``zinv``.

    A positive ``eps`` picks BART's regularised form, which divides by
    ``|x|^2 + eps`` rather than by ``x``.
    """

    _fn = "bartorch_nlop_zinv"

    def __init__(self, shape: Shape, eps: float = 0.0):
        self.eps = float(eps)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.eps,)

    def _bundle(self):
        from bartorch.nlop.base import chain
        from bartorch.nlop.bundle import diagonal, scaled

        # `zinv_reg_fun` squares the value and negates it, so the diagonal is
        # minus the square of the regularised inverse rather than of `1 / x`.
        squared = Multiply(self._shape, self._shape).dup(0, 1)
        made = chain(chain(Inverse(self._shape, self.eps), squared), scaled(self._shape, -1.0))
        return diagonal(self, made)

    def __repr__(self) -> str:
        return f"Inverse({self._shape}, eps={self.eps})"


class Power(_Elementwise):
    """``x ** exponent``, elementwise.  BART's ``zspow``."""

    _fn = "bartorch_nlop_zspow"

    def __init__(self, shape: Shape, exponent: complex):
        self.exponent = complex(exponent)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.exponent.real, self.exponent.imag)

    def _bundle(self):
        from bartorch.nlop.base import chain
        from bartorch.nlop.bundle import diagonal, scaled

        # `zspow_fun` divides the value by the point rather than raising the
        # point to one power less, which differs where the point is zero.
        # BART divides with `md_zdiv` and `Divide` multiplies by the inverse,
        # so this is that diagonal to within one rounding rather than to the
        # bit.
        ratio = chain(Power(self._shape, self.exponent), Divide(self._shape)).dup(0, 1)
        return diagonal(self, chain(ratio, scaled(self._shape, self.exponent)))

    def __repr__(self) -> str:
        return f"Power({self._shape}, {self.exponent})"


class Add(_Elementwise):
    """``x + value``, elementwise.  BART's ``zsadd``.

    Linear in name only: BART carries it as a nonlinear operator because an
    affine map is not one, and its derivative is the identity.
    """

    _fn = "bartorch_nlop_zsadd"

    def __init__(self, shape: Shape, value: complex):
        self.value = complex(value)
        super().__init__(shape)

    def _extra(self) -> tuple:
        return (self.value.real, self.value.imag)

    def _bundle(self):
        from bartorch.linop.basic import Identity
        from bartorch.nlop.base import FromLinear
        from bartorch.nlop.bundle import linear

        one = FromLinear(Identity(self._shape))
        return linear(self, one, one)

    def __repr__(self) -> str:
        return f"Add({self._shape}, {self.value})"


class _Reduction(NonlinearOperator):
    """One input reduced along ``axes``, which become one."""

    _fn: str = ""

    def __init__(self, shape: Shape, axes):
        self._shape = tuple(shape)
        self.axes = axes
        self._flags = axes_flags(axes, len(self._shape))
        self._out = tuple(
            1 if self._flags >> (len(self._shape) - 1 - axis) & 1 else size
            for axis, size in enumerate(self._shape)
        )
        super().__init__()

    def _extra(self) -> tuple:
        return ()

    def _create(self) -> Built:
        ptr = self._under_lock(
            getattr(library(), self._fn), DIMS, dims(self._shape), self._flags, *self._extra()
        )
        return Built(ptr, self._shape, self._out)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._shape}, axes={self.axes})"


class Sum(_Reduction):
    """``sum(|x|^2)`` along ``axes``.  BART's ``zss``."""

    _fn = "bartorch_nlop_zss"

    def _composition(self):
        """``nlop_zss_create``: the point against its own conjugate, summed, made real.

        The product contracts onto the reduced shape, as
        ``md_ztenmul`` does with a smaller output (``someops.c:539``).
        """
        from bartorch.linop.basic import Conj
        from bartorch.linop.shape import Real
        from bartorch.nlop.base import FromLinear, chain
        from bartorch.nlop.bundle import _TenMul

        product = chain(
            FromLinear(Conj(self._shape)),
            _TenMul(self._out, self._shape, self._shape),
            output=0,
            input=0,
        )
        return chain(product.dup(0, 1), FromLinear(Real(self._out)))

    def _bundle(self):
        from bartorch.nlop.bundle import of_composition

        return of_composition(self, self._composition())


class RootSumOfSquares(_Reduction):
    """``sqrt(sum(|x|^2))`` along ``axes``.  BART's ``zrss``.

    A positive ``eps`` picks BART's regularised form, whose derivative stays
    finite where the sum is zero.
    """

    _fn = "bartorch_nlop_zrss"

    def __init__(self, shape: Shape, axes, eps: float = 0.0):
        self.eps = float(eps)
        super().__init__(shape, axes)

    def _extra(self) -> tuple:
        return (self.eps,)

    def _composition(self):
        """``nlop_zrss_reg_create``: the sum of squares, offset where asked, rooted."""
        from bartorch.nlop.base import chain

        made = Sum(self._shape, self.axes)
        if self.eps:
            made = chain(made, Add(self._out, self.eps))
        return chain(made, Sqrt(self._out))

    def _bundle(self):
        from bartorch.nlop.bundle import of_composition

        return of_composition(self, self._composition())


class Multiply(NonlinearOperator):
    """The pointwise product of two inputs, broadcast where either is one.

    BART's ``tenmul``.  This is the operator every multi-argument model is
    built out of: ``nlinv``'s is an image of ``(1, X, Y)`` times coil profiles
    of ``(C, X, Y)``, and the derivative by either input is a multiplication
    by the other.

    Parameters
    ----------
    a, b : tuple of int
        The shape of each input, C order, of the same rank.  They broadcast
        the way torch's do.
    """

    def __init__(self, a: Shape, b: Shape):
        self._a, self._b = tuple(a), tuple(b)
        self._out = _broadcast(self._a, self._b)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_tenmul,
            DIMS,
            dims(self._out),
            dims(self._a),
            dims(self._b),
        )
        return _built(ptr, (self._a, self._b), (self._out,))

    def _bundle(self):
        from bartorch.nlop.bundle import of_product

        return of_product(self, self._out, self._a, self._b)

    def __repr__(self) -> str:
        return f"Multiply({self._a}, {self._b})"


class Divide(NonlinearOperator):
    """``a / b``, elementwise, of two inputs of one shape.  BART's ``zdiv``.

    A positive ``eps`` picks BART's regularised form.
    """

    def __init__(self, shape: Shape, eps: float = 0.0):
        self._shape = tuple(shape)
        self.eps = float(eps)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(library().bartorch_nlop_zdiv, DIMS, dims(self._shape), self.eps)
        return _built(ptr, (self._shape, self._shape), (self._shape,))

    def _composition(self):
        """``nlop_zdiv_reg_create``: the divisor inverted, then multiplied in."""
        from bartorch.nlop.base import chain

        return chain(
            Inverse(self._shape, self.eps),
            Multiply(self._shape, self._shape),
            output=0,
            input=1,
        )

    def _bundle(self):
        from bartorch.nlop.bundle import of_composition

        return of_composition(self, self._composition())

    def __repr__(self) -> str:
        return f"Divide({self._shape}, eps={self.eps})"


class Weighted(NonlinearOperator):
    """``a * x + b * z`` of two inputs of one shape.  BART's ``zaxpbz``."""

    def __init__(self, shape: Shape, a: float = 1.0, b: float = 1.0):
        self._shape = tuple(shape)
        self.a, self.b = float(a), float(b)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_zaxpbz, DIMS, dims(self._shape), self.a, self.b
        )
        return _built(ptr, (self._shape, self._shape), (self._shape,))

    def _bundle(self):
        from bartorch.nlop.base import combine
        from bartorch.nlop.bundle import linear, scaled

        # `a` and `b` are real, so the adjoint scales by them rather than by
        # their conjugates.
        adjoint = combine(scaled(self._shape, self.a), scaled(self._shape, self.b)).dup(0, 1)
        return linear(self, Weighted(self._shape, self.a, self.b), adjoint)

    def __repr__(self) -> str:
        return f"Weighted({self._shape}, a={self.a}, b={self.b})"


def Phase(shape: Shape) -> NonlinearOperator:  # noqa: N802
    """``x / |x|``, elementwise: BART's ``zphsr``, built out of its two pieces.

    BART's own ``nlop_zphsr_create`` chains ``zabs`` into ``zdiv`` and then
    calls ``nlop_dup(x, 0, 0)``, which trips its own ``a < b`` assertion, so
    the constructor aborts and cannot be used.  The same operator is these two
    pieces put together with the indices the algebra checks: the absolute
    value into the divisor, then the two inputs made one.
    """
    shape = tuple(shape)
    from bartorch.nlop.base import chain

    return chain(Abs(shape), Divide(shape), output=0, input=1).dup(0, 1)


class Constant(NonlinearOperator):
    """An operator of no inputs that returns ``value``.  BART's ``nlop_const``.

    Combined with another operator and linked into one of its inputs, this is
    how a model's fixed quantity -- a sampling pattern, an echo time -- is
    pinned; :meth:`~bartorch.nlop.NonlinearOperator.partial` does exactly that in
    one step.  BART copies the tensor, so the one passed in is free afterwards.
    """

    def __init__(self, value: torch.Tensor, shape: Shape | None = None):
        self.value = as_operand(value, tuple(shape) if shape is not None else value.shape, "value")
        super().__init__()

    def _create(self) -> Built:
        shape = tuple(self.value.shape)
        ptr = self._under_lock(
            library().bartorch_nlop_const,
            DIMS,
            dims(shape),
            self.value.data_ptr(),
        )
        return _built(ptr, (), (shape,))

    def _bundle(self):
        from bartorch.nlop.bundle import Bundle

        # No input, so no tangent to carry and no cotangent to return.
        return Bundle(self, Constant(torch.zeros_like(self.value)), None)

    def __repr__(self) -> str:
        return f"Constant({tuple(self.value.shape)})"
