"""BART's Gauss-Newton step, assembled over any model that supplies a bundle.

``noir_gauss_newton_step_create_s`` (``noir/model_net.c:361``) is one expression
in public ``nlop`` algebra over three operators of the linearization point.
:class:`Step` is that expression with a :class:`~bartorch.nlop.bundle.Bundle`
in their place; see ``docs/design/nonlinear-fusion.md``.
"""

from __future__ import annotations

import math

import torch

from bartorch._dispatch import BartError
from bartorch._lib import library
from bartorch._operator import Built, Shape
from bartorch.nlop.base import NonlinearOperator, _built, arity, chain
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


class Step(NonlinearOperator):
    """A Gauss-Newton schedule as one operator ``(y, xn, x0, alpha) -> x``.

    Each step is
    ``xn + (DF^H DF + alpha)^-1 [DF^H (y - F(xn)) - alpha (xn - x0)]``, with
    ``alpha`` a vector as long as the state and decaying by ``redu`` towards
    ``alpha_min``.  The state is the model's unknowns laid end to end;
    :meth:`split` and :meth:`join` read and write one.
    """

    def __init__(self, F, schedule, *, cg_lambda: float = 0.0):
        bundle = F.bundle
        if bundle is None:
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
        self.flat = flattened(F.bundle)
        self.iterations = int(schedule.iterations)
        self.redu = float(schedule.redu)
        self.alpha_min = float(schedule.alpha_min)
        self.cg_maxiter = int(schedule.cg_maxiter)
        self.cg_tol = float(schedule.cg_tol)
        self.cg_lambda = float(cg_lambda)
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
        data, state = self.data_shape, self.state_shape

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
        state = self.state_shape
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

    def _create(self) -> Built:
        # The outermost checkpoint is `model_net.c:423`, and it is also what
        # gives this operator a handle of its own rather than a second
        # reference to the assembly's.
        made = self._schedule()
        # BART holds the data at whatever rank the model's codomain has; the
        # caller passes what the model returns.
        made = made.reshape_input(0, self.data_shape)
        ptr = self._under_lock(
            library().bartorch_nlop_checkpoint, made._h.ptr, 1, 1, device=made.device
        )
        if not ptr:
            raise BartError("BART would not build the Gauss-Newton step")
        return _built(ptr, made.ishapes, made.oshapes, keep=(made,), device=made.device)

    # --- the arguments -----------------------------------------------------

    @property
    def data_shape(self) -> Shape:
        """What ``y`` is: whatever the model returns."""
        return self.flat.operator.oshapes[0]

    @property
    def state_shape(self) -> Shape:
        """What ``xn``, ``x0`` and the answer are: the unknowns laid end to end."""
        return self.flat.operator.ishapes[0]

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One state back into a tensor per unknown of the model."""
        flat, at = x.reshape(-1), 0
        out = []
        for shape in self.model.ishapes:
            size = math.prod(shape)
            out.append(flat[at : at + size].reshape(shape))
            at += size
        return tuple(out)

    def join(self, *xs: torch.Tensor) -> torch.Tensor:
        """One tensor per unknown laid end to end into a state."""
        return torch.cat([x.reshape(-1) for x in xs])

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
