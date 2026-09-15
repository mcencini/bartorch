"""The derivative of an operator with the linearization point as an argument.

``nlop_get_derivative`` gives a derivative at whatever point the last forward
left behind, which is state and so cannot be differentiated by.  A
:class:`Bundle` gives the same derivative with the point written out as an
argument, which is what a Gauss-Newton step is assembled over; see
``docs/design/nonlinear-fusion.md``.
"""

from __future__ import annotations

from functools import cached_property

import torch

from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, dims
from bartorch.nlop.base import FromLinear, NonlinearOperator, _built, chain, combine

__all__ = ["Bundle"]


class _TenMul(NonlinearOperator):
    """``md_ztenmul``: the product of two inputs, summed onto ``out``.

    :class:`~bartorch.nlop.Multiply` is this with ``out`` the broadcast of the
    two, which is what a forward model wants; an adjoint wants the axes the
    broadcast widened summed back instead.
    """

    def __init__(self, out: Shape, a: Shape, b: Shape):
        self._out, self._a, self._b = tuple(out), tuple(a), tuple(b)
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

    def __repr__(self) -> str:
        return f"_TenMul({self._out}, {self._a}, {self._b})"


def ignoring(op: NonlinearOperator, shapes) -> NonlinearOperator:
    """``op`` with one further input per shape, which nothing reads.

    A member that does not need the point still takes it, so that every
    bundle has the one arity.  The extra input reaches no output, so BART
    answers zero for the derivative by it, which is what the derivative of a
    point-independent member by the point is.
    """
    from bartorch.linop.basic import Zero

    made = op
    for shape in shapes:
        # A null map to one element rather than an identity, so the point is
        # never copied to be thrown away.
        made = combine(made, FromLinear(Zero((1,) * len(shape), shape)))
        made = made.del_out(len(made.oshapes) - 1)
    return made


def conjugate(op: NonlinearOperator) -> NonlinearOperator:
    """``conj(op(...))``, for the one-output case."""
    from bartorch.linop.basic import Conj

    return chain(op, FromLinear(Conj(op.oshape)), output=0, input=0)


def scaled(shape: Shape, value: complex) -> NonlinearOperator:
    """Multiplication by one number, as an operator of one input."""
    from bartorch.linop.base import _Scale

    return FromLinear(_Scale(value, tuple(shape)))


class Bundle:
    """``F``'s derivative and adjoint written with the linearization point as an argument.

    ``derivative`` takes ``(*tangents, *point)`` and returns one tangent per
    output; ``adjoint`` takes ``(*cotangents, *point)`` and returns one per
    input.  The tangent comes first because ``norm_inv_lambda_create`` reads
    input 0 as the vector it inverts over.

    Parameters
    ----------
    operator : NonlinearOperator
        Whose derivative this is; it is the bundle's ``forward``.
    derivative, adjoint : NonlinearOperator
        The two above.  ``adjoint`` is ``None`` only for an operator with no
        inputs, which has no cotangent to return.
    normal : NonlinearOperator, optional
        ``DF^H DF`` where the operator has a cheaper one than the two chained;
        by default they are chained, as ``noir_get_normal`` chains them.
    """

    def __init__(
        self,
        operator: NonlinearOperator,
        derivative: NonlinearOperator,
        adjoint: NonlinearOperator | None,
        normal: NonlinearOperator | None = None,
    ):
        self.operator = operator
        self.derivative = derivative
        self.adjoint = adjoint
        self._normal = normal
        self._check()

    @property
    def forward(self) -> NonlinearOperator:
        """``xn -> F(xn)``, which is the operator itself."""
        return self.operator

    def _check(self) -> None:
        ins, outs = self.operator.ishapes, self.operator.oshapes
        want = {
            "derivative": ((*ins, *ins), outs),
            "adjoint": ((*outs, *ins), ins),
        }
        for name, (ishapes, oshapes) in want.items():
            member = getattr(self, name)
            if member is None:
                if "adjoint" == name and not ins:
                    continue
                raise ValueError(f"a bundle needs a {name}")
            if member.ishapes != ishapes or member.oshapes != oshapes:
                raise ValueError(
                    f"{name} takes {member.ishapes} to {member.oshapes}, but a bundle for "
                    f"{type(self.operator).__name__} needs {ishapes} to {oshapes}"
                )

    @cached_property
    def normal(self) -> NonlinearOperator:
        """``(dx, xn) -> DF(xn)^H DF(xn) dx``, the two chained with the point shared."""
        if self._normal is not None:
            return self._normal
        if self.adjoint is None:
            raise NotImplementedError(
                f"{type(self.operator).__name__} has no inputs, so it has no normal operator"
            )
        if 1 != len(self.operator.oshapes):
            raise NotImplementedError(
                f"{type(self.operator).__name__} has {len(self.operator.oshapes)} outputs, and "
                "a normal operator is defined for one; take the outputs apart first"
            )
        n = len(self.operator.ishapes)
        # The chain leaves the adjoint's point in front of the derivative's
        # arguments, so the tangents are brought to the front and the two
        # copies of the point are made one, which is `noir_get_normal`.
        made = chain(self.derivative, self.adjoint, output=0, input=0)
        made = made.permute_inputs([*range(n, 2 * n), *range(n), *range(2 * n, 3 * n)])
        for at in range(n):
            made = made.dup(n + at, 2 * n)
        return made

    def __repr__(self) -> str:
        return f"Bundle({self.operator!r})"


def diagonal(operator: NonlinearOperator, diag: NonlinearOperator) -> Bundle:
    """The bundle of an elementwise operator whose derivative multiplies by ``diag(x)``.

    ``diag`` maps the point to the diagonal BART stores at the forward
    (``nlop_jacobian.c``), which it applies with ``md_ztenmul`` and whose
    conjugate it applies for the adjoint.
    """
    shape = operator.ishape
    return Bundle(
        operator,
        chain(diag, _TenMul(shape, shape, shape), output=0, input=1),
        chain(conjugate(diag), _TenMul(shape, shape, shape), output=0, input=1),
    )


def linear(operator: NonlinearOperator, forward, adjoint) -> Bundle:
    """The bundle of an operator linear in every input: the operator itself, point unused."""
    return Bundle(
        operator,
        ignoring(forward, operator.ishapes),
        ignoring(adjoint, operator.ishapes),
    )


def from_torch(operator: NonlinearOperator, fn) -> Bundle:
    """The bundle of a Python-defined operator, from torch's jvp and vjp at a given point.

    The members are themselves torch operators, so the step differentiating by
    the point differentiates ``fn`` a second time rather than reading a
    derivative this recorded.
    """
    from bartorch.nlop.callback import FromTorch

    ins, outs = operator.ishapes, operator.oshapes
    one_in, one_out = 1 == len(ins), 1 == len(outs)

    def derivative(*args):
        tangents, point = args[: len(ins)], args[len(ins) :]
        made = torch.func.jvp(fn, tuple(point), tuple(tangents))[1]
        return made if one_out else tuple(made)

    def adjoint(*args):
        cotangents, point = args[: len(outs)], args[len(outs) :]
        _, back = torch.func.vjp(fn, *point)
        made = back(cotangents[0] if one_out else tuple(cotangents))
        return made[0] if one_in else tuple(made)

    return Bundle(
        operator,
        FromTorch(derivative, [*ins, *ins], outs[0] if one_out else list(outs)),
        FromTorch(adjoint, [*outs, *ins], ins[0] if one_in else list(ins)),
    )
