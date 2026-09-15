"""BART's Gauss-Newton step, assembled over any model that supplies a bundle.

``noir_gauss_newton_step_create_s`` (``noir/model_net.c:361``) is one expression
in public ``nlop`` algebra over three operators of the linearization point.
:class:`Step` is that expression with a :class:`~bartorch.nlop.bundle.Bundle`
in their place; see ``docs/design/nonlinear-fusion.md``.
"""

from __future__ import annotations

import math

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError
from bartorch._lib import library
from bartorch._operator import Built, Shape
from bartorch.nlop import plan as _plan
from bartorch.nlop.base import NonlinearOperator, _bart_axis, _built, arity, chain
from bartorch.nlop.basic import Add, Multiply, Weighted
from bartorch.nlop.bundle import Bundle, scaled

__all__: list[str] = []


def swapped(a: NonlinearOperator, b: NonlinearOperator, *, output: int = 0, at: int = 0):
    """``nlop_chain2_swap``: a chain whose inputs put ``a``'s first.

    :func:`~bartorch.nlop.chain` leaves ``b``'s remaining inputs in front, and
    most of ``model_net.c``'s step is written the other way round.
    """
    made = chain(a, b, output=output, input=at)
    total, first = len(made.ishapes), len(a.ishapes)
    return made.permute_inputs([*range(total - first, total), *range(total - first)])


def _laid_out(op: NonlinearOperator, first: int, count: int, sizes) -> NonlinearOperator:
    """``count`` inputs from ``first`` flattened and laid end to end as one.

    ``nlop_flatten_in`` and then ``nlop_stack_inputs``, which is how
    ``noir_get_forward`` makes one vector of the image and the coils.
    """
    made = op
    for at in range(count):
        made = made.reshape_input(first + at, (sizes[at],))
    for _ in range(count - 1):
        made = made.stack_inputs(first, first + 1, 0)
    return made


def flattened(bundle: Bundle) -> Bundle:
    """``bundle`` over one flat vector of the model's unknowns.

    ``noir_gauss_newton_step_create_s`` asserts its state is one axis
    (``model_net.c:367``), so every model is written this way before a step is
    assembled over it -- one unknown or several, since what the assertion is
    about is the rank and not the count.  Laying shapes end to end moves no
    bytes.
    """
    op = bundle.operator
    n, m = len(op.ishapes), len(op.oshapes)
    sizes = [math.prod(shape) for shape in op.ishapes]

    # `nlop_flatten` would do this for the operator, but at BART's sixteen
    # axes, and the step asserts its state is one (`model_net.c:367`).  So each
    # member is written the way `noir_get_forward` writes itself: every
    # argument flattened on its own and then stacked.
    forward = _laid_out(op, 0, n, sizes)

    # The point first, so flattening it leaves the tangents where they are.
    derivative = _laid_out(_laid_out(bundle.derivative, n, n, sizes), 0, n, sizes)

    adjoint = _laid_out(bundle.adjoint, m, n, sizes)
    for at in range(n):
        adjoint = adjoint.reshape_output(at, (sizes[at],))
    for _ in range(n - 1):
        adjoint = adjoint.stack_outputs(0, 1, 0)

    return Bundle(forward, derivative, adjoint)


class _Built(NonlinearOperator):
    """One of the library's own constructors over operators already built."""

    def __init__(self, fn, args, *, keep, what: str):
        self.fn, self.args, self._keep, self.what = fn, args, keep, what
        super().__init__()

    def _create(self) -> Built:
        device = next((a.device for a in self._keep if a.device is not None), None)
        ptrs = tuple(a._h.ptr if isinstance(a, NonlinearOperator) else a for a in self.args)
        ptr = self._under_lock(self.fn, *ptrs, device=device)
        if not ptr:
            raise BartError(f"BART would not build {self.what}")
        ishapes, oshapes = arity(ptr)
        return _built(ptr, ishapes, oshapes, keep=self._keep, device=device)

    def __repr__(self) -> str:
        return f"<{self.what}>"


def flat_rank(op: NonlinearOperator) -> NonlinearOperator:
    """``op`` with every argument written at the rank its shape already has.

    The operators the step puts around the state are built at BART's sixteen
    axes here and at ``N = 1`` in ``model_net.c`` (`:380`, `:383`).  A reshape
    moves no bytes, and ``nlop_dup`` compares ``iovec``s rather than shapes, so
    without this the state and the step's own scratch refuse to meet.
    """
    made = op
    for at, shape in enumerate(op.ishapes):
        made = made.reshape_input(at, shape)
    for at, shape in enumerate(op.oshapes):
        made = made.reshape_output(at, shape)
    return made


def _checkpoint(x: NonlinearOperator, *, der_once: bool, clear_mem: bool) -> NonlinearOperator:
    return _Built(
        library().bartorch_nlop_checkpoint,
        (x, int(der_once), int(clear_mem)),
        keep=(x,),
        what="a checkpoint",
    )


class _Batched(NonlinearOperator):
    """``n`` builds of one operator applied together, stacked along a new leading axis.

    ``nlop_stack_multiple_F`` over one operator per item, which is how
    ``noir_gauss_newton_step_create`` (``model_net.c:425``) gives the noir step
    its batch.  The items share nothing -- each has its own build of the whole
    expression -- so one answers exactly what it would alone, and a solve does
    not couple them through the inner conjugate gradients.
    """

    #: BART's own flag for a stack whose parts are kept as a container.
    _CONTAINER = 1
    #: `nlop_stack_multiple_F`'s multigpu split, which needs a card to mean anything.
    _MULTIGPU = 0

    def __init__(self, copies):
        made = []
        for one in copies:
            # A singleton leading axis in C order is a singleton appended to
            # BART's dimension vector, which is what the stack is taken along.
            for at, shape in enumerate(one.ishapes):
                one = one.reshape_input(at, (1, *shape))
            for at, shape in enumerate(one.oshapes):
                one = one.reshape_output(at, (1, *shape))
            made.append(one._bart())
        self.copies = tuple(made)
        super().__init__()

    def _create(self) -> Built:
        first = self.copies[0]
        n = len(self.copies)
        handles = _marshal.handles([one._h.ptr for one in self.copies])
        instack = _marshal.ints([_bart_axis(0, shape) for shape in first.ishapes])
        outstack = _marshal.ints([_bart_axis(0, shape) for shape in first.oshapes])
        device = next((one.device for one in self.copies if one.device is not None), None)
        ptr = self._under_lock(
            library().bartorch_nlop_stack_multiple,
            n,
            handles,
            len(first.ishapes),
            instack,
            len(first.oshapes),
            outstack,
            self._CONTAINER,
            self._MULTIGPU,
            device=device,
        )
        if not ptr:
            raise BartError("BART would not stack the operators")
        ishapes = tuple((n, *shape[1:]) for shape in first.ishapes)
        oshapes = tuple((n, *shape[1:]) for shape in first.oshapes)
        return _built(ptr, ishapes, oshapes, keep=self.copies, device=device)

    def __repr__(self) -> str:
        return f"<{len(self.copies)} stacked>"


class Step(NonlinearOperator):
    """A Gauss-Newton schedule as one operator ``(y, xn, x0, alpha) -> x``.

    Each step is
    ``xn + (DF^H DF + alpha)^-1 [DF^H (y - F(xn)) - alpha (xn - x0)]``, with
    ``alpha`` a vector as long as the state and decaying by ``redu`` towards
    ``alpha_min``.  The state is the model's unknowns laid end to end;
    :meth:`split` and :meth:`join` read and write one.

    A coil composition is lowered so that its encoding is applied once as its
    normal, which moves ``y`` from samples to coil images; :meth:`prepare` puts
    a measurement there and :attr:`plan` says whether it happened.
    """

    def __init__(self, F, schedule, *, batch: int = 1, cg_lambda: float = 0.0, fuse: bool = True):
        if F.bundle is None:
            raise TypeError(
                f"{type(F).__name__} supplies no derivative as a function of the point, so a "
                "Gauss-Newton step cannot be assembled over it; IRGNM(...)(y, F, x0) solves "
                "with it instead"
            )
        if 1 != len(F.oshapes):
            raise ValueError(
                f"a step is assembled over a model with one output, and {type(F).__name__} has "
                f"{len(F.oshapes)}"
            )
        self.model = F
        #: What the model was lowered into, what prepares its data, and the plan.
        self.lowered, self._prepare, self.plan = _plan.build(F, fuse=fuse)
        self.flat = flattened(self.lowered.bundle)
        self.iterations = int(schedule.iterations)
        self.redu = float(schedule.redu)
        self.alpha_min = float(schedule.alpha_min)
        self.cg_maxiter = int(schedule.cg_maxiter)
        self.cg_tol = float(schedule.cg_tol)
        self.cg_lambda = float(cg_lambda)
        self.batch = int(batch)
        if 1 > self.batch:
            raise ValueError("a batch is at least one")
        super().__init__()

    # --- the assembly ------------------------------------------------------

    def _inverse(self) -> NonlinearOperator:
        """``(DF^H DF + alpha)^-1``, differentiated through the solve and not its steps."""
        normal = self.flat.normal
        return _Built(
            library().bartorch_nlop_norm_inv_lambda,
            (normal, self.cg_maxiter, self.cg_tol, self.cg_lambda),
            keep=(normal,),
            what="the inverse of the normal operator",
        )

    def _one_step(self) -> NonlinearOperator:
        """``noir_gauss_newton_step_create_s``, line for line, over the bundle."""
        data, state = self._item_data, self._item_state

        made = chain(self.flat.operator, Weighted(data, 1.0, -1.0), output=0, input=1)
        made = swapped(made, self.flat.adjoint)  # y, xn, xn
        made = made.dup(1, 2)  # y, xn
        made = swapped(made, flat_rank(Weighted(state, 1.0, -1.0)))  # y, xn, alpha (xn - x0)

        regulariser = swapped(
            flat_rank(Weighted(state, 1.0, -1.0)), flat_rank(Multiply(state, state))
        )
        made = chain(regulariser, made, output=0, input=2)  # y, xn, xn, x0, alpha
        made = made.dup(1, 2)  # y, xn, x0, alpha

        made = _checkpoint(made, der_once=True, clear_mem=True)

        made = swapped(made, self._inverse())  # y, xn, x0, alpha, xn, alpha
        made = made.dup(1, 4).dup(3, 4)  # y, xn, x0, alpha

        made = swapped(made, flat_rank(Weighted(state, 1.0, 1.0)))  # the step added
        return made.dup(1, 4)

    def _decay(self) -> NonlinearOperator:
        """``alpha -> (alpha - alpha_min) / redu + alpha_min``, as three of BART's pieces."""
        state = self._item_state
        made = chain(
            flat_rank(Add(state, -self.alpha_min)), flat_rank(scaled(state, 1.0 / self.redu))
        )
        return chain(made, flat_rank(Add(state, self.alpha_min)))

    def _schedule(self) -> NonlinearOperator:
        """``noir_gauss_newton_iter_create_s``: the steps, with the weight decaying between.

        Each further step is put in *front* of what is already assembled, so
        the weight the tail sees is the decayed one and the new step takes the
        argument as it stands.
        """
        made = _checkpoint(self._one_step(), der_once=False, clear_mem=True)
        for _ in range(self.iterations - 1):
            made = chain(self._decay(), made, output=0, input=3)
            step = _checkpoint(self._one_step(), der_once=False, clear_mem=True)
            made = swapped(step, made, output=0, at=1)
            made = made.dup(0, 4).dup(2, 4).dup(3, 4)
        return made

    def _assembled(self) -> NonlinearOperator:
        """One item's whole schedule, at the shapes a caller passes."""
        made = self._schedule()
        # BART holds the data at whatever rank the model's codomain has; the
        # caller passes what the model returns.
        return made.reshape_input(0, self._item_data)

    def _create(self) -> Built:
        # The outermost checkpoint is `model_net.c:423`, and it is also what
        # gives this operator a handle of its own rather than a second
        # reference to the assembly's.
        if 1 == self.batch:
            made = self._assembled()
        else:
            # `noir_gauss_newton_step_create` builds the whole step per item
            # and stacks the set, so each carries its own scratch and the
            # items share nothing.  The cost is linear in the batch, there as
            # here.
            made = _Batched([self._assembled() for _ in range(self.batch)])
        ptr = self._under_lock(
            library().bartorch_nlop_checkpoint, made._h.ptr, 1, 1, device=made.device
        )
        if not ptr:
            raise BartError("BART would not build the Gauss-Newton step")
        return _built(ptr, made.ishapes, made.oshapes, keep=(made,), device=made.device)

    # --- the arguments -----------------------------------------------------

    @property
    def _item_data(self) -> Shape:
        """One item's ``y``, which is what the assembly is built at."""
        return self.flat.operator.oshapes[0]

    @property
    def _item_state(self) -> Shape:
        """One item's state, which is what the assembly is built at."""
        return self.flat.operator.ishapes[0]

    def _batched(self, shape: Shape) -> Shape:
        return shape if 1 == self.batch else (self.batch, *shape)

    @property
    def data_shape(self) -> Shape:
        """What ``y`` is: what the model returns, or ``E^H`` of it once lowered.

        A batch is the leading axis, as it is for every other argument.
        """
        return self._batched(self._item_data)

    @property
    def state_shape(self) -> Shape:
        """What ``xn``, ``x0`` and the answer are: the unknowns laid end to end."""
        return self._batched(self._item_state)

    def prepare(self, y: torch.Tensor) -> torch.Tensor:
        """A measurement in the shape the step takes.

        ``E^H y`` where the step works in the normal-equation domain, which is
        what :attr:`plan` reports, and the measurement unchanged where it does
        not.  A batch is the leading axis and each item goes through the
        encoding on its own, the items sharing one encoding.
        """
        if self._prepare is None:
            return y
        if 1 == self.batch:
            return self._prepare.forward(y)
        return torch.stack([self._prepare.forward(one) for one in y])

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One state back into a tensor per unknown of the model, batch first."""
        flat = x.reshape(self.batch, -1) if 1 < self.batch else x.reshape(1, -1)
        out, at = [], 0
        for shape in self.lowered.ishapes:
            size = math.prod(shape)
            piece = flat[:, at : at + size]
            out.append(
                piece.reshape(self.batch, *shape) if 1 < self.batch else piece.reshape(shape)
            )
            at += size
        return tuple(out)

    def join(self, *xs: torch.Tensor) -> torch.Tensor:
        """One tensor per unknown laid end to end into a state, batch first."""
        if 1 == self.batch:
            return torch.cat([x.reshape(-1) for x in xs])
        return torch.cat([x.reshape(self.batch, -1) for x in xs], dim=1)

    def weight(self, alpha: float, device=None) -> torch.Tensor:
        """``alpha`` as the step takes it: a vector as long as the state, multiplied by."""
        return torch.full(self.state_shape, float(alpha), dtype=torch.complex64, device=device)

    def forward(self, *xs):
        """``(y, xn, x0, alpha)`` -> the iterate.  ``alpha`` may be a number."""
        if 4 == len(xs) and not isinstance(xs[3], torch.Tensor):
            reference = xs[1] if isinstance(xs[1], torch.Tensor) else None
            xs = (*xs[:3], self.weight(xs[3], None if reference is None else reference.device))
        return super().forward(*xs)

    def __repr__(self) -> str:
        return (
            f"IRGNM(iterations={self.iterations}, cg_maxiter={self.cg_maxiter})"
            f".operator({self.model!r})"
        )
