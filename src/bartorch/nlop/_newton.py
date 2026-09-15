"""BART's whole Gauss-Newton step of ``nlinv``, as :meth:`~bartorch.nlop.IRGNM.operator` builds it.

``noir/model_net.c`` assembles the step out of ``nlop``s throughout, so it
differentiates by its data, its iterate, its regularisation centre and its
weight; nothing here reimplements it.
"""

from __future__ import annotations

import math

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, _Handle, dims
from bartorch.nlop.base import NonlinearOperator, _built
from bartorch.nlop.mri import _optional

__all__: list[str] = []

#: BART's three spatial axes.
_FFT_FLAGS = 7


def _trim(shape: Shape) -> Shape:
    """A rank-sixteen BART shape as the rest of this library writes one.

    The leading axis is BART's batch -- its last, and so the first in C order
    -- and is kept whatever it is, one included, so that a batch of one and a
    batch of four have the same rank.  The run of singletons behind it is
    dropped, down to the first axis that holds something.
    """
    batch, rest = shape[0], list(shape[1:])
    while 1 < len(rest) and 1 == rest[0]:
        rest.pop(0)
    return (batch, *rest)


def _arity(ptr: int) -> tuple[tuple[Shape, ...], tuple[Shape, ...]]:
    """What BART says an operator takes and returns.

    Read rather than worked out: the iterate is a flat vector of the image and
    the coil coefficients together, whose length is BART's business, and the
    batch axis is BART's too.
    """
    lib = library()
    shapes: list[tuple[Shape, ...]] = []
    for count, query in (
        (lib.bartorch_nlop_inputs(ptr), lib.bartorch_nlop_input_domain),
        (lib.bartorch_nlop_outputs(ptr), lib.bartorch_nlop_output_codomain),
    ):
        each = []
        for at in range(count):
            vector = _marshal.wide_dim_vector()
            rank = query(ptr, at, len(vector), vector)
            if rank < 0:
                raise BartError("BART would not report the shape of one of its arguments")
            each.append(tuple(int(vector[i]) for i in range(rank))[::-1])
        shapes.append(tuple(each))
    return shapes[0], shapes[1]


class _Noir(NonlinearOperator):
    """BART's sixteen axes inside, and the shapes this library writes outside.

    :attr:`ishapes` and :attr:`oshapes` stay what BART reports, because that
    is what the arity check holds the operator to; :attr:`shapes` and
    :attr:`output_shapes` are the same tuples without the empty axes between
    the batch and the image, and are what a caller passes and gets back.  A
    run of singletons changes no strides, so converting between the two is a
    reshape rather than a copy.
    """

    @property
    def shapes(self) -> tuple[Shape, ...]:
        """What each argument is, without the axes BART leaves empty."""
        return tuple(_trim(s) for s in self.ishapes)

    @property
    def output_shapes(self) -> tuple[Shape, ...]:
        """The same for what comes back."""
        return tuple(_trim(s) for s in self.oshapes)

    @staticmethod
    def _fit(x, shape: Shape):
        """``x`` at ``shape``, when it is the same thing written shorter."""
        if not isinstance(x, torch.Tensor) or tuple(x.shape) == tuple(shape):
            return x
        if tuple(x.shape) == _trim(shape) or _trim(tuple(x.shape)) == tuple(shape):
            return x.reshape(shape)
        return x

    def forward(self, *xs):
        xs = tuple(self._fit(x, shape) for x, shape in zip(xs, self.ishapes))
        made = super().forward(*xs)
        if isinstance(made, torch.Tensor):
            return self._fit(made, self.output_shapes[0])
        return tuple(self._fit(one, shape) for one, shape in zip(made, self.output_shapes))


class _Companion(_Noir):
    """One of the model's other operators: the gridding, or a split."""

    def __init__(self, model, fn, what: str):
        self.model = model
        self.fn = fn
        self.what = what
        super().__init__()

    def forward(self, *xs):
        made = super().forward(*xs)
        if "prepare" == self.what:
            # `noir_adjoint_fft_fun` writes the pattern into the model on its
            # way past -- `linop_gdiag_set_diag(model->lop_pattern, ...)` --
            # and off the grid `noir_adjoint_nufft_fun` calls
            # `nufft_update_traj`.  That is how the model comes to have one.
            self.model._prepared = True
        return made

    def _create(self) -> Built:
        ptr = self._under_lock(self.fn, self.model._model.ptr, device=self.model.device)
        if not ptr:
            raise BartError(f"BART would not build the model's {self.what}")
        ishapes, oshapes = _arity(ptr)
        return _built(ptr, ishapes, oshapes, keep=(self.model._model,), device=self.model.device)

    def __repr__(self) -> str:
        return f"{self.model!r}.{self.what}()"


class _Cell(_Noir):
    """BART's whole Gauss-Newton step over the network model ``noir2_net``, built for a
    :class:`~bartorch.nlop.NonlinearSense` and an :class:`~bartorch.nlop.IRGNM`."""

    def __init__(self, sense, schedule, *, batch: int, cg_lambda: float):
        self.sense = sense
        self.iterations = int(schedule.iterations)
        self.redu = float(schedule.redu)
        self.alpha_min = float(schedule.alpha_min)
        self.cg_maxiter = int(schedule.cg_maxiter)
        self.cg_tol = float(schedule.cg_tol)
        self.cg_lambda = float(cg_lambda)
        self.batch = int(batch)
        self._model = self._make_model()
        # BART's model is built without a sampling pattern; `prepare` is what
        # gives it one, as a side effect of the gridding.  Applying a step
        # before that reads a diagonal nothing has written.
        self._prepared = False
        super().__init__()

    # --- the model ---------------------------------------------------------

    def _image_only(self) -> Shape:
        return self.sense._image_only()

    def _make_model(self) -> _Handle:
        _ensure_ready()
        lib = library()
        F = self.sense
        trj_dims, _ = _optional(F.trajectory)
        # BART calls the sampling pattern the model's weights, and off the
        # grid that is what it is: the density compensation.
        wgh_dims, _ = _optional(F.weights if F.noncart else F.pattern)
        bas_dims, basis = _optional(F.basis)
        msk_dims, mask = _optional(F.mask)
        a, b = F.sobolev

        held = (F.trajectory, F.pattern, F.weights, F.mask)
        self.device = next((t.device for t in held if t is not None), None)

        with _lock, _on_device(self.device or torch.device("cpu")):
            ptr = lib.bartorch_noir_net_create(
                DIMS,
                dims(F.kspace_shape),
                dims(F.coil_image_shape),
                dims(self._image_only()),
                dims(F.coil_shape),
                trj_dims,
                wgh_dims,
                bas_dims,
                basis,
                msk_dims,
                mask,
                0,
                self.batch,
                _FFT_FLAGS,
                _FFT_FLAGS,
                int(F.real),
                int(F.sos),
                a,
                b,
                F.c,
                int(F.toeplitz),
            )
        if not ptr:
            raise BartError("BART would not build the model; see the log for its message")
        return _Handle(ptr, lib.bartorch_noir_net_free, ())

    def _create(self) -> Built:
        lib = library()
        with _lock, _on_device(self.device or torch.device("cpu")):
            ptr = lib.bartorch_noir_net_iterations(
                self._model.ptr,
                self.cg_maxiter,
                self.cg_tol,
                self.cg_lambda,
                self.iterations,
                self.redu,
                self.alpha_min,
            )
        if not ptr:
            raise BartError("BART would not build the Gauss-Newton step")
        ishapes, oshapes = _arity(ptr)
        return _built(ptr, ishapes, oshapes, keep=(self._model,), device=self.device)

    # --- the arguments -----------------------------------------------------

    @property
    def data_shape(self) -> Shape:
        """What ``y`` is: coil images, batch first."""
        return self.shapes[0]

    @property
    def state_shape(self) -> Shape:
        """What ``xn``, ``x0`` and the answer are: the image and the coil
        coefficients laid end to end, batch first."""
        return self.shapes[1]

    def start(self, batch: int | None = None, device=None) -> torch.Tensor:
        """The iterate ``nlinv`` starts from, as ``noir2_init`` writes it: ones, and no coils."""
        made = torch.zeros(self.state_shape, dtype=torch.complex64, device=device)
        made[..., : math.prod(self._image_only())] = 1.0
        if batch is not None and batch != made.shape[0]:
            made = made[:1].expand(batch, *made.shape[1:]).contiguous()
        return made

    def weight(self, alpha: float, device=None) -> torch.Tensor:
        """``alpha`` as the operator takes it: a vector as long as the iterate, multiplied by."""
        return torch.full(self.state_shape, float(alpha), dtype=torch.complex64, device=device)

    def forward(self, *xs):
        """``(y, xn, x0, alpha)`` -> the iterate.  ``alpha`` may be a number."""
        if not self._prepared:
            raise BartError(
                "this operator's model has no sampling pattern yet: BART writes one into the "
                "model as a side effect of the gridding, so `prepare()` has to be applied to "
                "*this* operator before a step is.  Without it the step reads a diagonal that "
                "was never written, which is a segmentation fault and not an error.  Two "
                "operators do not share a model, so each needs its own `prepare()`"
            )
        if 4 == len(xs) and not isinstance(xs[3], torch.Tensor):
            reference = xs[1] if isinstance(xs[1], torch.Tensor) else None
            xs = (*xs[:3], self.weight(xs[3], None if reference is None else reference.device))
        return super().forward(*xs)

    # --- the model's other operators ---------------------------------------

    def prepare(self) -> NonlinearOperator:
        """``(kspace, pattern)`` -> the data a step takes; off the grid the pattern is the weights.

        Applying it is also what gives this operator's model its pattern.
        """
        return _Companion(self, library().bartorch_noir_net_adjoint, "prepare")

    def decompose(self) -> NonlinearOperator:
        """``x`` -> ``(image, sensitivities)``, through the model's transforms."""
        return _Companion(self, library().bartorch_noir_net_decompose, "decompose")

    def split(self) -> NonlinearOperator:
        """``x`` -> ``(image, coefficients)``, with no transform applied."""
        return _Companion(self, library().bartorch_noir_net_split, "split")

    def join(self) -> NonlinearOperator:
        """``(image, coefficients)`` -> ``x``: :meth:`split` the other way."""
        return _Companion(self, library().bartorch_noir_net_join, "join")

    def __repr__(self) -> str:
        return (
            f"IRGNM(iterations={self.iterations}, cg_maxiter={self.cg_maxiter})"
            f".operator({self.sense!r}, batch={self.batch})"
        )
