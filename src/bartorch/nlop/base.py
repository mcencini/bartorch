"""The nonlinear operator base class and composition.

BART's ``nlop_s`` is a map from many inputs to many outputs, and the algebra
in ``nlops/chain.h`` is what puts several of them together: an output into an
input, two operators side by side, an output tied back to an input, two
inputs made one.  The model ``nlinv`` inverts is two inputs, the image and
the coil profiles, so none of it is reachable from a one-in one-out wrapper.

Arguments are counted BART's way throughout -- outputs first, then inputs --
which is the order every index below and every buffer handed to BART is in.
"""

from __future__ import annotations

import math
from functools import cached_property

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Operator, Shape, _Handle, as_operand, dims

__all__ = ["Chain", "FromLinear", "NonlinearOperator", "chain", "combine"]


def _built(
    ptr: int,
    ishapes: tuple[Shape, ...],
    oshapes: tuple[Shape, ...],
    *,
    keep: tuple = (),
    device: torch.device | None = None,
) -> Built:
    """A :class:`~bartorch._operator.Built` of any arity.

    ``ishape`` and ``oshape`` stay filled in for the one-to-one case, which is
    all that most of the library ever asks an operator for.
    """
    one = lambda shapes: shapes[0] if 1 == len(shapes) else ()  # noqa: E731
    return Built(
        ptr,
        one(ishapes),
        one(oshapes),
        keep=keep,
        device=device,
        ishapes=tuple(tuple(s) for s in ishapes),
        oshapes=tuple(tuple(s) for s in oshapes),
    )


def _index(i: int, n: int, what: str) -> int:
    """One argument index, counted from the end when negative."""
    at = i + n if i < 0 else i
    if not 0 <= at < n:
        raise IndexError(f"{what} {i} is out of range for an operator with {n} of them")
    return at


def _bart_axis(axis: int, shape: Shape) -> int:
    """The BART dimension a C-order axis of ``shape`` is."""
    at = axis + len(shape) if axis < 0 else axis
    if not 0 <= at < len(shape):
        raise IndexError(f"axis {axis} is out of range for shape {shape}")
    return len(shape) - 1 - at


class NonlinearOperator(Operator):
    """A map between shapes, with a derivative and its adjoint.

    :meth:`derivative` and :meth:`adjoint` are taken at the point of the last
    :meth:`forward` call, which is how BART's solvers use them.  The backward
    pass of ``F(x)`` is ``adjoint`` at that point, so evaluating the operator
    elsewhere between a forward and a backward pass gives a wrong gradient.

    An operator may take more than one input and return more than one output;
    :attr:`ishapes` and :attr:`oshapes` are the shape of each, and
    :attr:`ishape` and :attr:`oshape` are the whole of it when there is one of
    each.  The algebra in :mod:`bartorch.nlop` is what builds the many-argument
    ones: :func:`combine`, :func:`chain`, :meth:`link`, :meth:`dup`.

    A subclass is defined either by :meth:`_create`, which builds one of
    BART's operators, or in Python by :meth:`forward`, :meth:`derivative` and
    :meth:`adjoint`.

    Attributes
    ----------
    ishapes, oshapes : tuple of tuple of int
        The shape of each input and of each output, C order.
    ishape, oshape : tuple of int
        Domain and codomain, for an operator with one input and one output.
    """

    _free_name = "bartorch_nlop_free"
    _domain_name = "bartorch_nlop_domain"
    _codomain_name = "bartorch_nlop_codomain"

    #: The shape of each argument, outputs and inputs kept apart.
    ishapes: tuple[Shape, ...] = ()
    oshapes: tuple[Shape, ...] = ()

    def __init__(self):
        if self._native:
            self._build()
        elif any(
            getattr(type(self), name) is getattr(NonlinearOperator, name)
            for name in ("forward", "derivative", "adjoint")
        ):
            raise TypeError(
                f"{type(self).__name__} must define _create, or forward, derivative and adjoint"
            )

    # --- shape ------------------------------------------------------------

    def _only(self, shapes: tuple[Shape, ...], what: str) -> Shape:
        if 1 != len(shapes):
            raise ValueError(
                f"{type(self).__name__} has {len(shapes)} {what}s, so it has no single "
                f"{what[0]}shape; its {what}s are {what[0]}shapes"
            )
        return shapes[0]

    @property
    def ishape(self) -> Shape:
        """The domain, for an operator with one input."""
        return self._only(self.ishapes, "input")

    @property
    def oshape(self) -> Shape:
        """The codomain, for an operator with one output."""
        return self._only(self.oshapes, "output")

    def _build(self) -> None:
        _ensure_ready()
        built = self._create()
        lib = library()
        self._h = _Handle(built.ptr, lib.bartorch_nlop_free, built.keep)
        self.device = built.device
        ishapes = built.ishapes if built.ishapes is not None else (tuple(built.ishape),)
        oshapes = built.oshapes if built.oshapes is not None else (tuple(built.oshape),)
        self.ishapes = tuple(tuple(s) for s in ishapes)
        self.oshapes = tuple(tuple(s) for s in oshapes)
        self._check_shapes()

    def _check_shapes(self) -> None:
        """Raise unless BART agrees on the arity and on every argument's shape.

        The combinators work out what they produce from what they were given,
        which is how a user's own ranks survive them; this is what holds that
        bookkeeping to BART's.
        """
        lib = library()
        for query, shapes, what in (
            (lib.bartorch_nlop_input_domain, self.ishapes, "input"),
            (lib.bartorch_nlop_output_codomain, self.oshapes, "output"),
        ):
            count = (lib.bartorch_nlop_inputs if "input" == what else lib.bartorch_nlop_outputs)(
                self._h.ptr
            )
            if count != len(shapes):
                raise BartError(
                    f"the operator has {count} {what}s in BART but {len(shapes)} were recorded"
                )
            for at, shape in enumerate(shapes):
                vector = _marshal.dim_vector()
                if query(self._h.ptr, at, DIMS, vector) < 0:
                    raise BartError(f"BART would not report the shape of {what} {at}")
                bart = [int(vector[i]) for i in range(DIMS)]
                if bart != list(dims(shape)):
                    raise BartError(f"{what} {at} is {bart[::-1]} in BART but {shape} was recorded")

    # --- evaluation order --------------------------------------------------

    @cached_property
    def _stages(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Which evaluation stage each output and each input belongs to.

        BART evaluates a combination back to front: ``operator_combi_create``
        applies its operands in reverse, so in ``combine(a, b)`` it is ``b``
        that runs first.  That is why ``nlop_chain2`` combines ``b`` with ``a``
        and not the other way round, and it is why linking an output into an
        input that is consumed earlier reads a buffer nothing has written yet
        -- silently, with no complaint from BART.

        Lower runs first.  A single operator is one stage, and the
        combinators compose these the way BART composes the operators.
        """
        return (0,) * len(self.oshapes), (0,) * len(self.ishapes)

    # --- application ------------------------------------------------------

    def forward(self, *xs: torch.Tensor):
        """``F(x)``, which also fixes where every derivative is taken.

        Takes one tensor per input and returns one per output, or the tensor
        itself when there is a single output.
        """
        if 1 == len(self.ishapes) == len(self.oshapes) and 1 == len(xs):
            return self._apply(library().bartorch_nlop_apply, xs[0], self.ishape, self.oshape)
        return self._apply_generic(xs)

    def derivative(self, dx: torch.Tensor) -> torch.Tensor:
        """``DF(x) dx`` at the last evaluated point, for one input and one output."""
        return self._apply(library().bartorch_nlop_derivative, dx, self.ishape, self.oshape)

    def adjoint(self, dy: torch.Tensor) -> torch.Tensor:
        """``DF(x)^H dy`` at the last evaluated point, for one input and one output."""
        return self._apply(library().bartorch_nlop_adjoint, dy, self.oshape, self.ishape)

    def _apply_generic(self, xs):
        """Apply an operator of any arity through ``nlop_generic_apply_unchecked``."""
        if not self._native:
            raise NotImplementedError(
                f"{type(self).__name__} takes {len(self.ishapes)} inputs and "
                f"{len(self.oshapes)} outputs, which only a BART-backed operator does"
            )
        if len(xs) != len(self.ishapes):
            raise ValueError(f"the operator takes {len(self.ishapes)} inputs, got {len(xs)}")
        ins = [
            as_operand(x, shape, f"input {at}")
            for at, (x, shape) in enumerate(zip(xs, self.ishapes))
        ]
        devices = {x.device for x in ins}
        if 1 < len(devices):
            raise ValueError(f"the inputs are on more than one device: {sorted(map(str, devices))}")
        device = ins[0].device if ins else torch.device("cpu")
        outs = [torch.empty(s, dtype=torch.complex64, device=device) for s in self.oshapes]
        # Outputs first, then inputs, which is the order BART reads them in.
        buffers = [t.data_ptr() for t in outs] + [t.data_ptr() for t in ins]
        args = _marshal.pointers(buffers)
        with _lock, _on_device(self.device or device):
            if 0 != library().bartorch_nlop_apply_generic(self._h.ptr, len(buffers), args):
                raise BartError("operator application failed; see the log for BART's message")
        return outs[0] if 1 == len(outs) else tuple(outs)

    def _as_callbacks(self) -> NonlinearOperator:
        from bartorch.nlop.callback import Callback

        if 1 != len(self.ishapes) or 1 != len(self.oshapes):
            raise NotImplementedError(
                "a Python-defined operator of more than one argument has no BART callback "
                "behind it; build it from one-argument pieces with the algebra instead"
            )
        return Callback(self.oshape, self.ishape, self.forward, self.derivative, self.adjoint)

    # --- the derivative as a linear operator -------------------------------

    def jacobian(self, output: int = 0, input: int = 0):  # noqa: A002
        """``DF/dx_input`` of one output, as a :class:`~bartorch.linop.LinearOperator`.

        The whole linear surface applies to what this returns -- its adjoint,
        its normal operator, a solve over it -- which is how ``noir/recon2.c``
        builds the inner problem of a Gauss-Newton step.

        The point is wherever the last :meth:`forward` left it, and it moves
        with the next one: this is a view of the operator's derivative, not a
        copy of it.
        """
        from bartorch.nlop.derivative import Derivative

        return Derivative(self, output, input)

    def linearize(self, *xs: torch.Tensor):
        """The derivative at ``x``, as a :class:`~bartorch.linop.LinearOperator`.

        Evaluates the operator at ``x``, which moves the point the derivative
        is taken at.  For an operator of one input and one output; for the
        others, evaluate and take a :meth:`jacobian`.
        """
        from bartorch.linop.basic import Callback as LinearCallback

        self.forward(*xs)
        return LinearCallback(self.oshape, self.ishape, self.derivative, self.adjoint)

    def __call__(self, *xs: torch.Tensor):
        """``F(x)``, recorded for autograd when an input requires a gradient."""
        tracking = (
            any(isinstance(x, torch.Tensor) and x.requires_grad for x in xs)
            and torch.is_grad_enabled()
        )
        if tracking:
            from bartorch.nlop.autograd import apply

            return apply(self, *xs)
        return self.forward(*xs)

    # --- the algebra -------------------------------------------------------

    def link(self, output: int = 0, input: int = 0) -> NonlinearOperator:  # noqa: A002
        """Tie an output back into an input; both arguments go away.

        ``nlop_link``.  What is left is ``f(..., g(...), ...)`` with the
        intermediate no longer visible.
        """
        return _Link(self, output, input)

    def dup(self, a: int = 0, b: int = 1) -> NonlinearOperator:
        """Make two inputs of the same shape one input, kept at ``a``.

        ``nlop_dup``.  The derivative by the surviving input is the sum of the
        two, which is what makes this the way to tie a parameter to itself.
        """
        return _Dup(self, a, b)

    def stack_inputs(self, a: int, b: int, axis: int) -> NonlinearOperator:
        """Make two inputs one, concatenated along a C-order ``axis``.

        ``nlop_stack_inputs``.  The stacked input takes the lower of the two
        positions; the shapes must agree away from ``axis``.
        """
        return _Stack(self, a, b, axis, inputs=True)

    def stack_outputs(self, a: int, b: int, axis: int) -> NonlinearOperator:
        """Make two outputs one, concatenated along a C-order ``axis``.

        ``nlop_stack_outputs``.
        """
        return _Stack(self, a, b, axis, inputs=False)

    def permute_inputs(self, perm) -> NonlinearOperator:
        """Reorder the inputs: the new input ``i`` is the old ``perm[i]``."""
        return _Permute(self, perm, outputs=False)

    def permute_outputs(self, perm) -> NonlinearOperator:
        """Reorder the outputs: the new output ``o`` is the old ``perm[o]``."""
        return _Permute(self, perm, outputs=True)

    def shift_input(self, new: int, old: int) -> NonlinearOperator:
        """Move one input to another position, the rest closing up behind it."""
        return self.permute_inputs(_shifted(len(self.ishapes), new, old))

    def shift_output(self, new: int, old: int) -> NonlinearOperator:
        """Move one output to another position, the rest closing up behind it."""
        return self.permute_outputs(_shifted(len(self.oshapes), new, old))

    def pin(self, input: int, value) -> NonlinearOperator:  # noqa: A002
        """Fix one input to ``value``; the input goes away.

        ``nlop_set_input_const``.  What a model's fixed quantities are -- an
        echo time, a sampling pattern -- once the operator that takes them has
        been built.  BART copies the tensor, so the one passed in is free
        afterwards.
        """
        return _Pinned(self, input, value)

    def del_out(self, output: int = 0) -> NonlinearOperator:
        """Drop an output, and everything computed only for it.

        ``nlop_del_out``.
        """
        return _DelOut(self, output)

    def flatten(self, inputs_only: bool = False) -> NonlinearOperator:
        """Every input as one flat vector, and every output as another.

        ``nlop_flatten``.  What a two-unknown model needs to reach a solver
        that knows one vector: ``noir/recon2.c`` lays the image and the coil
        coefficients out one after the other, in argument order, and that is
        what :class:`~bartorch.optim.IRGNM` is handed.
        """
        return _Flattened(self, inputs_only)

    def combine(self, other: NonlinearOperator) -> NonlinearOperator:
        """``self`` and ``other`` side by side, sharing nothing."""
        return combine(self, other)

    def chain(
        self,
        other: NonlinearOperator,
        *,
        output: int = 0,
        input: int = 0,  # noqa: A002
    ) -> NonlinearOperator:
        """Output ``output`` of ``self`` into input ``input`` of ``other``."""
        return chain(self, other, output=output, input=input)

    def __matmul__(self, other):
        """``self @ other`` applies ``other`` first, as one BART operator."""
        from bartorch.linop.base import LinearOperator

        if isinstance(other, LinearOperator):
            other = other.to_nonlinear()
        if not isinstance(other, NonlinearOperator):
            return NotImplemented
        return Chain(self, other)

    def __rmatmul__(self, other):
        from bartorch.linop.base import LinearOperator

        if not isinstance(other, LinearOperator):
            return NotImplemented
        return Chain(other.to_nonlinear(), self)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"{self.ishapes[0] if 1 == len(self.ishapes) else list(self.ishapes)} -> "
            f"{self.oshapes[0] if 1 == len(self.oshapes) else list(self.oshapes)})"
        )


def _shifted(n: int, new: int, old: int) -> tuple[int, ...]:
    """The permutation that moves argument ``old`` to position ``new``."""
    rest = [i for i in range(n) if i != _index(old, n, "argument")]
    at = _index(new, n, "argument")
    return tuple(rest[:at] + [_index(old, n, "argument")] + rest[at:])


class FromLinear(NonlinearOperator):
    """A linear operator as a nonlinear one, whose derivative is itself."""

    def __init__(self, op):
        self.op = op._bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_from_linop, self.op._h.ptr, device=self.op.device
        )
        return Built(ptr, self.op.ishape, self.op.oshape, keep=(self.op,), device=self.op.device)

    def __repr__(self) -> str:
        return f"{self.op!r}.to_nonlinear()"


class Chain(NonlinearOperator):
    """``a @ b`` as one BART operator; ``b`` is applied first."""

    def __init__(self, a: NonlinearOperator, b: NonlinearOperator):
        self.a, self.b = a._bart(), b._bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_chain,
            self.b._h.ptr,
            self.a._h.ptr,
            device=self.a.device or self.b.device,
        )
        return Built(
            ptr,
            self.b.ishape,
            self.a.oshape,
            keep=(self.a, self.b),
            device=self.a.device or self.b.device,
        )

    def __repr__(self) -> str:
        return f"({self.a!r} @ {self.b!r})"


class _Binary(NonlinearOperator):
    """Two operators made one; the subclass says which of BART's calls does it."""

    def __init__(self, a: NonlinearOperator, b: NonlinearOperator):
        if a is b:
            raise ValueError(
                "an operator cannot be put together with itself: each holds the point its "
                "derivative is taken at, and the two would overwrite one another"
            )
        self.a, self.b = a._bart(), b._bart()
        super().__init__()

    @property
    def _device(self):
        return self.a.device or self.b.device


class _Combine(_Binary):
    """``nlop_combine``: two operators side by side."""

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_combine, self.a._h.ptr, self.b._h.ptr, device=self._device
        )
        return _built(
            ptr,
            self.a.ishapes + self.b.ishapes,
            self.a.oshapes + self.b.oshapes,
            keep=(self.a, self.b),
            device=self._device,
        )

    @cached_property
    def _stages(self):
        ao, ai = self.a._stages
        bo, bi = self.b._stages
        # BART applies the second operand first.
        ao, ai = _after((ao, ai), (bo, bi))
        return ao + bo, ai + bi

    def __repr__(self) -> str:
        return f"combine({self.a!r}, {self.b!r})"


class _Chain2(_Binary):
    """``nlop_chain2``: one output of ``a`` into one input of ``b``."""

    def __init__(self, a, b, output: int, input: int):  # noqa: A002
        self.output = _index(output, len(a.oshapes), "output")
        self.input = _index(input, len(b.ishapes), "input")
        if a.oshapes[self.output] != b.ishapes[self.input]:
            raise ValueError(
                f"output {self.output} is {a.oshapes[self.output]} and input {self.input} "
                f"is {b.ishapes[self.input]}; a chain needs them to agree"
            )
        super().__init__(a, b)

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_chain2,
            self.a._h.ptr,
            self.output,
            self.b._h.ptr,
            self.input,
            device=self._device,
        )
        # BART combines b with a and links across, which leaves b's inputs
        # without the one that was fed, then a's; and b's outputs, then a's
        # without the one that fed it.
        ishapes = _without(self.b.ishapes, self.input) + self.a.ishapes
        oshapes = self.b.oshapes + _without(self.a.oshapes, self.output)
        return _built(ptr, ishapes, oshapes, keep=(self.a, self.b), device=self._device)

    @cached_property
    def _stages(self):
        ao, ai = self.a._stages
        bo, bi = self.b._stages
        # nlop_chain2 combines b with a, so a runs first, as it must.
        bo, bi = _after((bo, bi), (ao, ai))
        return bo + _without(ao, self.output), _without(bi, self.input) + ai

    def __repr__(self) -> str:
        return f"chain({self.a!r}, {self.b!r}, output={self.output}, input={self.input})"


def _without(shapes: tuple, at: int) -> tuple:
    return shapes[:at] + shapes[at + 1 :]


def _after(stages: tuple[tuple[int, ...], ...], first: tuple[tuple[int, ...], ...]):
    """``stages``, moved past everything in ``first``."""
    seen = [s for group in first for s in group]
    shift = max(seen) + 1 if seen else 0
    return tuple(tuple(s + shift for s in group) for group in stages)


class _Unary(NonlinearOperator):
    """One operator rearranged."""

    def __init__(self, x: NonlinearOperator):
        self.x = x._bart()
        super().__init__()


class _Link(_Unary):
    """``nlop_link``: an output tied back into an input."""

    def __init__(self, x, output: int, input: int):  # noqa: A002
        self.output = _index(output, len(x.oshapes), "output")
        self.input = _index(input, len(x.ishapes), "input")
        if x.oshapes[self.output] != x.ishapes[self.input]:
            raise ValueError(
                f"output {self.output} is {x.oshapes[self.output]} and input {self.input} "
                f"is {x.ishapes[self.input]}; a link needs them to agree"
            )
        produced, consumed = x._stages[0][self.output], x._stages[1][self.input]
        if produced >= consumed:
            raise ValueError(
                f"output {self.output} is produced after input {self.input} is read, so the "
                "link would read a buffer nothing has written yet.  BART applies a "
                "combination back to front, so the operator that produces goes second in "
                "combine(); chain() puts them in that order for you"
            )
        super().__init__(x)

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_link,
            self.x._h.ptr,
            self.output,
            self.input,
            device=self.x.device,
        )
        return _built(
            ptr,
            _without(self.x.ishapes, self.input),
            _without(self.x.oshapes, self.output),
            keep=(self.x,),
            device=self.x.device,
        )

    @cached_property
    def _stages(self):
        o, i = self.x._stages
        return _without(o, self.output), _without(i, self.input)

    def __repr__(self) -> str:
        return f"{self.x!r}.link({self.output}, {self.input})"


class _Dup(_Unary):
    """``nlop_dup``: two inputs made one."""

    def __init__(self, x, a: int, b: int):
        self.a = _index(a, len(x.ishapes), "input")
        self.b = _index(b, len(x.ishapes), "input")
        if self.a >= self.b:
            raise ValueError(f"dup takes the inputs in order, got {self.a} and {self.b}")
        if x.ishapes[self.a] != x.ishapes[self.b]:
            raise ValueError(
                f"input {self.a} is {x.ishapes[self.a]} and input {self.b} is "
                f"{x.ishapes[self.b]}; only inputs of one shape can be made one"
            )
        super().__init__(x)

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_dup, self.x._h.ptr, self.a, self.b, device=self.x.device
        )
        return _built(
            ptr,
            _without(self.x.ishapes, self.b),
            self.x.oshapes,
            keep=(self.x,),
            device=self.x.device,
        )

    @cached_property
    def _stages(self):
        o, i = self.x._stages
        kept = list(i)
        kept[self.a] = min(i[self.a], i[self.b])
        return o, _without(tuple(kept), self.b)

    def __repr__(self) -> str:
        return f"{self.x!r}.dup({self.a}, {self.b})"


class _Stack(_Unary):
    """``nlop_stack_inputs`` and ``nlop_stack_outputs``: two arguments concatenated."""

    def __init__(self, x, a: int, b: int, axis: int, *, inputs: bool):
        self.inputs = inputs
        shapes = x.ishapes if inputs else x.oshapes
        what = "input" if inputs else "output"
        self.a = _index(a, len(shapes), what)
        self.b = _index(b, len(shapes), what)
        if self.a == self.b:
            raise ValueError(f"{what} {self.a} cannot be stacked with itself")
        first, second = shapes[self.a], shapes[self.b]
        if len(first) != len(second):
            raise ValueError(f"{what}s {first} and {second} do not have the same rank")
        self.axis = axis + len(first) if axis < 0 else axis
        if not 0 <= self.axis < len(first):
            raise IndexError(f"axis {axis} is out of range for shape {first}")
        if _without(tuple(first), self.axis) != _without(tuple(second), self.axis):
            raise ValueError(
                f"{what}s {first} and {second} differ away from axis {self.axis}, "
                "so there is no shape to stack them into"
            )
        self.stacked = (
            first[: self.axis] + (first[self.axis] + second[self.axis],) + first[self.axis + 1 :]
        )
        super().__init__(x)

    def _create(self) -> Built:
        shapes = self.x.ishapes if self.inputs else self.x.oshapes
        fn = (
            library().bartorch_nlop_stack_inputs
            if self.inputs
            else library().bartorch_nlop_stack_outputs
        )
        ptr = self._under_lock(
            fn,
            self.x._h.ptr,
            self.a,
            self.b,
            _bart_axis(self.axis, shapes[self.a]),
            device=self.x.device,
        )
        # The stacked argument takes the lower of the two positions.
        low, high = min(self.a, self.b), max(self.a, self.b)
        left = _without(_without(shapes, high), low)
        made = left[:low] + (self.stacked,) + left[low:]
        ishapes = made if self.inputs else self.x.ishapes
        oshapes = self.x.oshapes if self.inputs else made
        return _built(ptr, ishapes, oshapes, keep=(self.x,), device=self.x.device)

    @cached_property
    def _stages(self):
        o, i = self.x._stages
        stages = i if self.inputs else o
        low, high = min(self.a, self.b), max(self.a, self.b)
        # The stacking operator itself runs alongside; the earlier of the two
        # is what a later link has to clear.
        left = _without(_without(stages, high), low)
        made = left[:low] + (min(stages[self.a], stages[self.b]),) + left[low:]
        return (o, made) if self.inputs else (made, i)

    def __repr__(self) -> str:
        what = "stack_inputs" if self.inputs else "stack_outputs"
        return f"{self.x!r}.{what}({self.a}, {self.b}, {self.axis})"


class _Permute(_Unary):
    """``nlop_permute_inputs`` and ``nlop_permute_outputs``."""

    def __init__(self, x, perm, *, outputs: bool):
        self.outputs = outputs
        shapes = x.oshapes if outputs else x.ishapes
        what = "output" if outputs else "input"
        perm = tuple(_index(p, len(shapes), what) for p in perm)
        if sorted(perm) != list(range(len(shapes))):
            raise ValueError(
                f"a permutation of {len(shapes)} {what}s takes each of them once, got {perm}"
            )
        self.perm = perm
        super().__init__(x)

    def _create(self) -> Built:
        shapes = self.x.oshapes if self.outputs else self.x.ishapes
        vector = _marshal.ints(self.perm)
        ptr = self._under_lock(
            library().bartorch_nlop_permute,
            self.x._h.ptr,
            1 if self.outputs else 0,
            len(self.perm),
            vector,
            device=self.x.device,
        )
        moved = tuple(shapes[p] for p in self.perm)
        ishapes = self.x.ishapes if self.outputs else moved
        oshapes = moved if self.outputs else self.x.oshapes
        return _built(ptr, ishapes, oshapes, keep=(self.x,), device=self.x.device)

    @cached_property
    def _stages(self):
        o, i = self.x._stages
        moved = tuple((o if self.outputs else i)[p] for p in self.perm)
        return (moved, i) if self.outputs else (o, moved)

    def __repr__(self) -> str:
        what = "permute_outputs" if self.outputs else "permute_inputs"
        return f"{self.x!r}.{what}({list(self.perm)})"


class _DelOut(_Unary):
    """``nlop_del_out``: an output dropped."""

    def __init__(self, x, output: int):
        self.output = _index(output, len(x.oshapes), "output")
        if 1 == len(x.oshapes):
            raise ValueError("dropping the only output would leave nothing to compute")
        super().__init__(x)

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_del_out, self.x._h.ptr, self.output, device=self.x.device
        )
        return _built(
            ptr,
            self.x.ishapes,
            _without(self.x.oshapes, self.output),
            keep=(self.x,),
            device=self.x.device,
        )

    @cached_property
    def _stages(self):
        o, i = self.x._stages
        return _without(o, self.output), i

    def __repr__(self) -> str:
        return f"{self.x!r}.del_out({self.output})"


class _Flattened(_Unary):
    """``nlop_flatten``: every argument laid out end to end in one vector."""

    def __init__(self, x, inputs_only: bool):
        self.inputs_only = bool(inputs_only)
        self.sizes = tuple(math.prod(s) for s in x.ishapes)
        self.out_sizes = tuple(math.prod(s) for s in x.oshapes)
        super().__init__(x)

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_nlop_flatten,
            self.x._h.ptr,
            int(self.inputs_only),
            device=self.x.device,
        )
        oshapes = self.x.oshapes if self.inputs_only else ((sum(self.out_sizes),),)
        return _built(ptr, ((sum(self.sizes),),), oshapes, keep=(self.x,), device=self.x.device)

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One flat vector back into a tensor per input of the operator it flattened."""
        if math.prod(tuple(x.shape)) != sum(self.sizes):
            raise ValueError(f"expected {sum(self.sizes)} values, got {tuple(x.shape)}")
        flat = x.reshape(-1)
        out, at = [], 0
        for shape, size in zip(self.x.ishapes, self.sizes):
            out.append(flat[at : at + size].reshape(shape))
            at += size
        return tuple(out)

    @cached_property
    def _stages(self):
        o, i = self.x._stages
        least = lambda group: (min(group),) if group else ()  # noqa: E731
        return (o if self.inputs_only else least(o)), least(i)

    def __repr__(self) -> str:
        return f"{self.x!r}.flatten()"


class _Pinned(_Unary):
    """``nlop_set_input_const``: one input fixed to a tensor."""

    def __init__(self, x, input: int, value):  # noqa: A002
        self.input = _index(input, len(x.ishapes), "input")
        self.value = as_operand(value, x.ishapes[self.input], f"the value for input {self.input}")
        super().__init__(x)

    def _create(self) -> Built:
        shape = self.x.ishapes[self.input]
        ptr = self._under_lock(
            library().bartorch_nlop_set_input_const,
            self.x._h.ptr,
            self.input,
            DIMS,
            dims(shape),
            self.value.data_ptr(),
            device=self.x.device,
        )
        return _built(
            ptr,
            _without(self.x.ishapes, self.input),
            self.x.oshapes,
            keep=(self.x,),
            device=self.x.device,
        )

    @cached_property
    def _stages(self):
        o, i = self.x._stages
        return o, _without(i, self.input)

    def __repr__(self) -> str:
        return f"{self.x!r}.pin({self.input}, ...)"


def combine(a: NonlinearOperator, b: NonlinearOperator) -> NonlinearOperator:
    """``a`` and ``b`` side by side, sharing nothing.

    ``nlop_combine``.  The result takes ``a``'s inputs and then ``b``'s, and
    returns ``a``'s outputs and then ``b``'s.  It is where every other
    combination starts: put two operators next to each other, then
    :meth:`~NonlinearOperator.link` or :meth:`~NonlinearOperator.dup` the
    arguments that are meant to be the same.

    BART applies a combination back to front -- ``b`` runs first -- so put
    whichever operator produces a value second, and the one that consumes it
    first.  That is the order ``nlop_chain2`` builds, and :func:`chain` takes
    care of it; :meth:`~NonlinearOperator.link` refuses the other way round
    rather than reading a buffer nothing has written.
    """
    return _Combine(a, b)


def chain(
    a: NonlinearOperator,
    b: NonlinearOperator,
    *,
    output: int = 0,
    input: int = 0,  # noqa: A002
) -> NonlinearOperator:
    """Output ``output`` of ``a`` into input ``input`` of ``b``.

    ``nlop_chain2``.  ``a`` runs first.  The result takes ``b``'s remaining
    inputs and then all of ``a``'s, and returns all of ``b``'s outputs and then
    ``a``'s remaining ones.

    For two operators of one argument each this is ``b @ a``, which keeps the
    reading order of composition; this is the form that reaches the rest.
    """
    return _Chain2(a, b, output, input)
