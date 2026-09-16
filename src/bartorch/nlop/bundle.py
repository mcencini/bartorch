"""The derivative of an operator with the linearization point as an argument.

``nlop_get_derivative`` gives a derivative at whatever point the last forward
left behind, which is state and so cannot be differentiated by.  A
:class:`Bundle` gives the same derivative with the point written out as an
argument, over which a Gauss-Newton step is assembled; see
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
    two, as a forward model requires; an adjoint requires the axes the
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

    def _bundle(self) -> Bundle:
        return of_product(self, self._out, self._a, self._b)

    def __repr__(self) -> str:
        return f"_TenMul({self._out}, {self._a}, {self._b})"


def ignoring(op: NonlinearOperator, shapes) -> NonlinearOperator:
    """``op`` with one further input per shape, which nothing reads.

    A member that does not need the point still takes it, so that every
    bundle has the one arity.  The extra input reaches no output, so BART
    answers zero for the derivative by it, which is the derivative of a
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
    """Derivative and adjoint of a nonlinear operator as functions of the linearization point.

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
    source : str
        Where the bundle came from, as reported by a model's ``plan``:
        ``"declared"``, ``"linear"``, ``"torch"`` or ``"chain rule"``.
    """

    def __init__(
        self,
        operator: NonlinearOperator,
        derivative: NonlinearOperator,
        adjoint: NonlinearOperator | None,
        normal: NonlinearOperator | None = None,
        source: str = "declared",
    ):
        self.operator = operator
        self.derivative = derivative
        self.adjoint = adjoint
        self.source = source
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

    def at(self, point):
        """``DF(point)`` as a linear operator that holds its point.

        Applications do not depend on what was evaluated before them, and are
        differentiable by ``point``.  For an operator of one input and one output.
        """
        from bartorch.nlop.derivative import Linearization

        return Linearization(self, point)

    def __repr__(self) -> str:
        return f"Bundle({self.operator!r})"


class Asymmetric(FromLinear):
    """A linear stage whose bundle's adjoint is not the adjoint of its forward.

    ``noir2_join`` builds the non-Cartesian model's last stage as
    ``linop_from_ops(lop_fft->normal, identity->adjoint)`` (``model2.c:176``):
    the forward applies ``E^H E`` and the adjoint applies nothing, the
    measurement having already been through ``E^H``.  That pair is not one linear
    operator's forward and adjoint, so it is declared here rather than built as
    one.
    """

    def __init__(self, forward, adjoint, source=None):
        self.backward = adjoint._bart()
        #: The encoding the forward is the normal of, where one is known.
        self.source = None if source is None else source._bart()
        super().__init__(forward)

    def _bundle(self) -> Bundle:
        return linear(self, FromLinear(self.op), FromLinear(self.backward))

    def __repr__(self) -> str:
        return f"Asymmetric({self.op!r}, {self.backward!r})"


def _only_output(op: NonlinearOperator, at: int) -> NonlinearOperator:
    """``op`` with every output but ``at`` dropped, which leaves its inputs alone."""
    made = op
    for output in reversed(range(len(op.oshapes))):
        if output != at:
            made = made.del_out(output)
    return made


def of_chain(node, first, second, output: int, at: int) -> Bundle | None:
    """The chain rule for ``first``'s output ``output`` feeding ``second``'s input ``at``.

    The point ``second`` is linearized at is ``first(x)``, recomputed from the
    point rather than carried, as ``noir_get_derivative`` does with
    the coil model's linear parts.  ``None`` where either operand has no
    bundle, or where ``first`` takes no input and so returns no cotangent.
    """
    one, two = first.bundle, second.bundle
    if one is None or two is None or one.adjoint is None or two.adjoint is None:
        return None

    na, ma = len(first.ishapes), len(first.oshapes)
    nb, mb = len(second.ishapes), len(second.oshapes)
    if 1 > na:
        return None

    # `first` appears in both members, and in the normal in one graph twice.
    # Both copies are handed the same point by construction -- that is what
    # the duplications below are for -- so the derivative BART stores in it is
    # the same either way.
    made = chain(one.derivative, two.derivative, output=output, input=at)
    made = chain(_only_output(first, output), made, output=0, input=(nb - 1) + at)
    base = 2 * nb - 2
    for j in range(na):
        made = made.dup(base + na + j, base + 2 * na)
    derivative = made.permute_inputs(
        [
            *range(nb - 1),
            *range(base, base + na),
            *range(nb - 1, base),
            *range(base + na, base + 2 * na),
        ]
    )

    p, q = ma - 1, ma - 1 + na
    r, s = q + mb, q + mb + nb - 1
    made = chain(two.adjoint, one.adjoint, output=at, input=output)
    made = chain(_only_output(first, output), made, output=0, input=p + na + mb + at)
    for j in range(na):
        made = made.dup(p + j, s)
    adjoint = made.permute_inputs([*range(q, r), *range(p), *range(r, s), *range(p, q)])
    # The chain returns `first`'s cotangents first and the node's inputs put
    # `second`'s first, so the outputs are read the other way round.
    adjoint = adjoint.permute_outputs([*range(na, na + nb - 1), *range(na)])

    return Bundle(node, derivative, adjoint, source="chain rule")


def of_combine(node, a, b) -> Bundle | None:
    """The chain rule for two operators side by side: block diagonal in both members."""
    one, two = a.bundle, b.bundle
    if one is None or two is None or one.adjoint is None or two.adjoint is None:
        return None

    na, ma = len(a.ishapes), len(a.oshapes)
    nb, mb = len(b.ishapes), len(b.oshapes)

    # Each member lays its own tangents and point end to end, and the node
    # wants every tangent before every point.
    derivative = combine(one.derivative, two.derivative).permute_inputs(
        [
            *range(na),
            *range(2 * na, 2 * na + nb),
            *range(na, 2 * na),
            *range(2 * na + nb, 2 * na + 2 * nb),
        ]
    )
    adjoint = combine(one.adjoint, two.adjoint).permute_inputs(
        [
            *range(ma),
            *range(ma + na, ma + na + mb),
            *range(ma, ma + na),
            *range(ma + na + mb, ma + na + mb + nb),
        ]
    )
    return Bundle(node, derivative, adjoint, source="chain rule")


def of_dup(node, x, a: int, b: int) -> Bundle | None:
    """The chain rule for two inputs made one: the tangent paths add, the point is shared."""
    from bartorch.nlop.basic import Weighted

    inner = x.bundle
    if inner is None or inner.adjoint is None:
        return None

    n, m = len(x.ishapes), len(x.oshapes)
    # The later of a pair goes, so the point's pair is merged first and the
    # tangents' own indices do not move under it.
    derivative = inner.derivative.dup(n + a, n + b).dup(a, b)

    made = inner.adjoint.dup(m + a, m + b)
    # Both cotangents survive `nlop_dup`, and the one the merged input takes
    # is their sum.
    made = chain(made, Weighted(x.ishapes[a], 1.0, 1.0), output=a, input=0)
    # The sum leads the outputs and the other cotangent has moved up one.
    made = made.link(b, 0)

    rest = [at for at in range(n) if at not in (a, b)]
    target = [at for at in range(n) if at != b]
    adjoint = made.permute_outputs([0 if at == a else 1 + rest.index(at) for at in target])
    return Bundle(node, derivative, adjoint, source="chain rule")


def of_permute(node, x, perm, *, outputs: bool) -> Bundle | None:
    """The chain rule for reordered arguments: the tangents move with them."""
    inner = x.bundle
    if inner is None or inner.adjoint is None:
        return None

    n, m = len(x.ishapes), len(x.oshapes)
    if outputs:
        derivative = inner.derivative.permute_outputs(perm)
        adjoint = inner.adjoint.permute_inputs([*perm, *range(m, m + n)])
    else:
        derivative = inner.derivative.permute_inputs([*perm, *(n + p for p in perm)])
        adjoint = inner.adjoint.permute_inputs([*range(m), *(m + p for p in perm)]).permute_outputs(
            perm
        )
    return Bundle(node, derivative, adjoint, source="chain rule")


def of_reshape(node, x, at: int, shape: Shape, *, output: bool) -> Bundle | None:
    """The chain rule for one argument written at another rank, which moves no bytes."""
    inner = x.bundle
    if inner is None or inner.adjoint is None:
        return None

    n, m = len(x.ishapes), len(x.oshapes)
    if output:
        derivative = inner.derivative.reshape_output(at, shape)
        adjoint = inner.adjoint.reshape_input(at, shape)
    else:
        derivative = inner.derivative.reshape_input(at, shape).reshape_input(n + at, shape)
        adjoint = inner.adjoint.reshape_input(m + at, shape).reshape_output(at, shape)
    return Bundle(node, derivative, adjoint, source="chain rule")


def of_pinned(node, x, at: int, value) -> Bundle | None:
    """The chain rule for an input held fixed: no tangent of its own, and the point is the value."""
    inner = x.bundle
    if inner is None or inner.adjoint is None:
        return None

    n, m = len(x.ishapes), len(x.oshapes)
    zero = torch.zeros_like(value)
    # The point first, so pinning it does not move the tangent's index.
    derivative = inner.derivative.partial(n + at, value).partial(at, zero)
    adjoint = inner.adjoint.partial(m + at, value).del_out(at)
    return Bundle(node, derivative, adjoint, source="chain rule")


def of_del_out(node, x, at: int) -> Bundle | None:
    """The chain rule for a dropped output: it carries no tangent and its cotangent is zero."""
    inner = x.bundle
    if inner is None or inner.adjoint is None:
        return None

    derivative = inner.derivative.del_out(at)
    zero = torch.zeros(x.oshapes[at], dtype=torch.complex64)
    adjoint = inner.adjoint.partial(at, zero)
    return Bundle(node, derivative, adjoint, source="chain rule")


def of_product(node, out: Shape, a: Shape, b: Shape) -> Bundle:
    """The product rule for a tensor product of two inputs.

    ``noir_get_derivative``'s two terms added (``model_net.c:299-315``), and
    ``noir_get_adjoint``'s pair with each operand conjugated and the product
    summed back onto the other's shape (``:269-287``).
    """
    from bartorch.linop.basic import Conj
    from bartorch.nlop.basic import Weighted

    made = combine(_TenMul(out, a, b), _TenMul(out, a, b))  # in: a, db, da, b
    made = made.permute_inputs([2, 1, 0, 3])  # in: da, db, a, b
    derivative = chain(made, Weighted(out, 1.0, 1.0), output=0, input=0).link(1, 0)

    first = chain(FromLinear(Conj(b)), _TenMul(a, b, out), output=0, input=0)
    second = chain(FromLinear(Conj(a)), _TenMul(b, a, out), output=0, input=0)
    adjoint = combine(first, second)  # in: dz, b, dz, a
    adjoint = adjoint.permute_inputs([0, 2, 3, 1]).dup(0, 1)  # in: dz, a, b

    return Bundle(node, derivative, adjoint)


def of_composition(node, written) -> Bundle | None:
    """The bundle of an operator declared as the composition BART builds it from.

    ``written`` is that composition, and the bundle is the chain rule's over it;
    this only relabels the members as the operator's own, so that a caller sees
    the operator it asked about rather than the pieces.  ``None`` where the
    composition does not answer for the operator or has no bundle itself.
    """
    if written.ishapes != node.ishapes or written.oshapes != node.oshapes:
        return None
    inner = written.bundle
    if inner is None:
        return None
    return Bundle(node, inner.derivative, inner.adjoint, source=inner.source)


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
        source="linear",
    )


def from_torch(operator: NonlinearOperator, fn) -> Bundle:
    """The bundle of a Python-defined operator, from torch's jvp and vjp at a given point.

    The members are themselves torch operators, so the step differentiating by
    the point differentiates ``fn`` a second time rather than reading a
    derivative this recorded.
    """
    from bartorch.nlop.callback import TorchOperator

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
        TorchOperator(derivative, [*ins, *ins], outs[0] if one_out else list(outs)),
        TorchOperator(adjoint, [*outs, *ins], ins[0] if one_in else list(ins)),
        source="torch",
    )
