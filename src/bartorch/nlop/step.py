"""What a Gauss-Newton step applies, for a model that supplies a bundle.

``noir_gauss_newton_step_create_s`` (``noir/model_net.c:361``) applies three
operators of the linearization point: the model, its adjoint derivative, and the
inverse of its normal operator plus a weight.  :class:`Linearized` holds the
three for one model, lowered as :mod:`bartorch.nlop.plan` decides, with what
prepares the data and lays the unknowns end to end;
:class:`~bartorch.nlop.IRGNMBlock` applies them.  See
``docs/design/nonlinear-fusion.md``.
"""

from __future__ import annotations

import math

import torch

from bartorch._dispatch import BartError
from bartorch._lib import library
from bartorch._operator import Built, Shape
from bartorch.nlop import plan as _plan
from bartorch.nlop.base import NonlinearOperator, _built, arity
from bartorch.nlop.bundle import Bundle

__all__: list[str] = []


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


def _inverse(normal, maxiter: int, tol: float, l2lambda: float) -> NonlinearOperator:
    """``(b, xn, alpha) -> (DF^H DF + alpha)^-1 b``, differentiated through the solve."""
    return _Built(
        library().bartorch_nlop_norm_inv_lambda,
        (normal, int(maxiter), float(tol), float(l2lambda)),
        keep=(normal,),
        what="the inverse of the normal operator",
    )


class Linearized:
    """A model prepared for Gauss-Newton steps.

    ``operator``, ``adjoint`` and :meth:`inverse` are the three operators a step
    applies, over the unknowns laid end to end.  A coil composition is lowered
    so that its encoding is applied once as its normal, which moves the data
    from samples to coil images; :meth:`prepare` puts a measurement there and
    :attr:`plan` says whether it happened.  A leading batch axis is an item
    per entry, each prepared, split and joined on its own.
    """

    def __init__(self, F, *, cg_maxiter=30, cg_tol=0.0, cg_lambda=0.0, fuse=True, inverse=True):
        if F.bundle is None:
            raise TypeError(
                f"{type(F).__name__} supplies no derivative as a function of the point, so no "
                "Gauss-Newton step can be taken over it"
            )
        if 1 != len(F.oshapes):
            raise ValueError(
                f"a step is taken over a model with one output, and {type(F).__name__} has "
                f"{len(F.oshapes)}"
            )
        self.model = F
        #: What the model was lowered into, what prepares its data, and the plan.
        self.lowered, self._prepare, self.plan = _plan.build(F, fuse=fuse)
        self.flat = flattened(self.lowered.bundle)
        self._solve = (cg_maxiter, cg_tol, cg_lambda) if inverse else None
        self._inverses: dict = {}

    @property
    def operator(self) -> NonlinearOperator:
        """``xn -> F(xn)``, over the flat state."""
        return self.flat.operator

    @property
    def adjoint(self) -> NonlinearOperator:
        """``(r, xn) -> DF(xn)^H r``, over the flat state."""
        return self.flat.adjoint

    def inverse(self, device=None) -> NonlinearOperator:
        """``(b, xn, alpha) -> (DF^H DF + alpha)^-1 b``, one per device.

        ``norm_inv`` keeps copies of its arguments where its first application
        put them, so an inverse applied on the host is not applied on a card.
        """
        if self._solve is None:
            raise ValueError("this model was prepared for an inner solver, with no inverse")
        key = str(torch.device(device)) if device is not None else "cpu"
        made = self._inverses.get(key)
        if made is None:
            made = self._inverses[key] = _inverse(self.flat.normal, *self._solve)
        return made

    @property
    def data_shape(self) -> Shape:
        """One item's data: what the model returns, or ``E^H`` of it once lowered."""
        return tuple(self.flat.operator.oshapes[0])

    @property
    def state_shape(self) -> Shape:
        """One item's state: the model's unknowns laid end to end."""
        return tuple(self.flat.operator.ishapes[0])

    def batched(self, x: torch.Tensor, shape) -> bool:
        """Whether ``x`` carries a batch axis in front of ``shape``."""
        return x.ndim == len(shape) + 1

    def prepare(self, y: torch.Tensor) -> torch.Tensor:
        """A measurement at the data shape: ``E^H y`` where the plan says the normal domain."""
        model = tuple(self.model.oshapes[0])

        def one(item):
            made = item if self._prepare is None else self._prepare.forward(item)
            return made.reshape(self.data_shape)

        return torch.stack([one(v) for v in y]) if self.batched(y, model) else one(y)

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """A state back into a tensor per unknown of the model, batch first."""
        batch = x.shape[:1] if self.batched(x, self.state_shape) else ()
        flat = x.reshape(*batch, -1)
        out, at = [], 0
        for shape in self.lowered.ishapes:
            size = math.prod(shape)
            out.append(flat[..., at : at + size].reshape(*batch, *shape))
            at += size
        return tuple(out)

    def join(self, *xs: torch.Tensor) -> torch.Tensor:
        """A tensor per unknown laid end to end into a state, batch first."""
        first = self.lowered.ishapes[0]
        batch = xs[0].shape[:1] if xs[0].ndim == len(first) + 1 else ()
        return torch.cat([x.reshape(*batch, -1) for x in xs], dim=-1)

    def state(self, x) -> torch.Tensor:
        """``x`` as a state: a tuple is joined, a tensor laid flat, batch first.

        A tensor is either the state itself or, for a model of one unknown, that
        unknown at its own shape.
        """
        if isinstance(x, (tuple, list)):
            return self.join(*x)
        shapes = [self.state_shape]
        if 1 == len(self.lowered.ishapes):
            shapes.append(tuple(self.lowered.ishapes[0]))
        for shape in shapes:
            if tuple(x.shape) == shape:
                return x.reshape(self.state_shape)
            if tuple(x.shape[1:]) == shape:
                return x.reshape(x.shape[0], *self.state_shape)
        raise ValueError(
            f"a state is {self.state_shape}, or the unknown at its own shape, with an optional "
            f"batch in front; not {tuple(x.shape)}"
        )

    def start(self, batch=(), device=None) -> torch.Tensor:
        """The iterate ``nlinv`` starts from: the first unknown ones, the rest zero.

        ``noir2_init``'s convention -- an image of ones and no coil
        coefficients.
        """
        made = torch.zeros((*batch, *self.state_shape), dtype=torch.complex64, device=device)
        made[..., : math.prod(self.lowered.ishapes[0])] = 1.0
        return made
