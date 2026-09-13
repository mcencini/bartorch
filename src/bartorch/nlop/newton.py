"""BART's Gauss-Newton step, which is already an operator with a derivative.

``noir/model_net.c`` builds one iteration of ``nlinv`` as an ``nlop``:

    x_{n+1} = x_n + (DF^H DF + alpha)^-1 [ DF^H (y - F(x_n)) - alpha (x_n - x_0) ]

and it builds it out of ``nlop``s throughout -- the forward model, the
*derivative as a function of the linearisation point*, the adjoint, and
``norm_inv``'s implicitly differentiated inverse of the normal operator.  So
the step has a derivative of its own, with respect to the data, the iterate,
the regularisation centre and the weight, second-order terms included.  That
is the cell an unrolled NLINV is made of, and BART reconstructs with it in
``networks/nlinvnet.c``.

What is here is that operator, not a reimplementation of it.  The arithmetic
is the library's; so is the gradient checkpointing inside the unrolled form,
which is what keeps a long unroll's memory down.

The model is the one :class:`~bartorch.nlop.NonlinearSense` wraps, built for a
network instead: a batch of independent copies, and the sampling pattern and
trajectory handed over per call rather than held.  ``bartorch`` has no other
operator whose batch is BART's own -- everywhere else a leading axis is
applied item by item from Python -- and here it is, because ``nlinvnet``
needed it.

Examples
--------
One step, differentiated to the data:

>>> newton = nlop.GaussNewton((coils, 256, 256), batch=4)
>>> y = newton.prepare()(kspace, pattern)
>>> x0 = newton.start(batch=4)
>>> x1 = newton(y, x0, x0, 1.0)

The whole of ``nlinv``, as one operator that trains:

>>> newton = nlop.GaussNewton((coils, 256, 256), iterations=8, redu=2.0)
>>> image, coils = newton.decompose()(newton(y, x0, x0, 1.0))
"""

from __future__ import annotations

import math

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, _Handle, dims
from bartorch.linop.mri import _spatial
from bartorch.nlop.base import NonlinearOperator, _built
from bartorch.nlop.mri import _optional, _tensor

__all__ = ["GaussNewton"]

#: BART's three spatial axes.
_FFT_FLAGS = 7


def _trim(shape: Shape) -> Shape:
    """A rank-sixteen BART shape as the rest of this library writes one.

    Everything here carries BART's batch axis, which is its *last* and so the
    leading one in C order; that stays whatever it is, one included, or a
    batch of one and a batch of four would be different ranks.  What goes is
    the run of singletons behind it, down to the first axis that holds
    something -- the same trimming :class:`~bartorch.nlop.NonlinearSense`
    does by writing its shapes out.
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
            rank = query(ptr, at, DIMS + 1, vector)
            if rank < 0:
                raise BartError("BART would not report the shape of one of its arguments")
            each.append(tuple(int(vector[i]) for i in range(rank))[::-1])
        shapes.append(tuple(each))
    return shapes[0], shapes[1]


class _Noir(NonlinearOperator):
    """BART's sixteen axes inside, and the shapes this library writes outside.

    These operators are stacked over BART's *batch* axis, which is its last
    and so the leading one in C order.  The axes between it and the image are
    empty, and writing them out is the difference between ``(4, 2, 1, 8, 8)``
    and a tuple with eleven ones in the middle of it -- the same memory
    either way, since a run of singletons changes no strides, so moving
    between the two is a reshape and not a copy.

    What BART reports stays the recorded shape, because that is what the
    arity check holds the operator to.  What a caller passes and what comes
    back is the short form, and :attr:`shapes` says what it is.
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
            # and the non-Cartesian one updates the trajectory the same way.
            # That is how the model comes to have one at all.
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


class GaussNewton(_Noir):
    """One or more Gauss-Newton steps of ``nlinv``, as an operator that differentiates.

    Takes ``(y, xn, x0, alpha)`` and returns the iterate:

    ``y``
        The data as the step takes it -- coil images, which :meth:`prepare`
        makes from k-space and a sampling pattern.
    ``xn``
        The iterate: the image and the coil coefficients in one flat vector.
        :meth:`start` makes the one BART starts from, and :meth:`decompose`
        reads one back.
    ``x0``
        The centre the step regularises towards, which ``nlinv`` leaves at the
        starting point.
    ``alpha``
        The Tikhonov weight.  A number is taken as one; with ``iterations``
        above one it is the *initial* weight, divided by ``redu`` after each
        step and never taken below ``alpha_min``, which is what
        :class:`~bartorch.optim.IRGNM` does.

    Every one of those carries a gradient, which is what makes this an
    unrolled network's cell rather than a solver's inside.  A denoiser between
    two of these -- or a term in the regularisation centre -- is NLINV-Net.

    Parameters
    ----------
    image_shape : tuple of int
        Coil-image shape, ``(coils, *spatial)``, as the linear encodings take
        it.
    pattern : tensor, optional
        Only to give the shape BART should expect; the pattern itself is an
        argument of :meth:`prepare`, because a network is handed one per call.
        Ones over the image by default.
    trajectory : tensor, optional
        Off the grid.  Its shape is what is kept, for the same reason.
    iterations : int
        Steps.  One is a step; more is that many, with the weight decaying,
        wrapped in BART's own gradient checkpointing.
    redu : float
        What the weight is divided by after each step.
    alpha_min : float
        What the weight decays towards.
    batch : int
        Independent copies of the model, stacked on BART's batch axis.  This
        is the leading axis of every argument.
    cg_maxiter, cg_tol, cg_lambda : int, float, float
        The conjugate gradients inside each step: ``iter_conjgrad_conf``'s
        ``maxiter``, ``tol`` and ``l2lambda``.
    weights, basis, mask : tensor, optional
        As :class:`~bartorch.nlop.NonlinearSense` takes them.  These *are*
        held by the model.
    sobolev : tuple of float
        ``(a, b)`` of the coil weighting ``(1 + a |k|^2)^(-b/2)``.
    real : bool
        Constrain the image to be real (``nlinv -R``).
    sos : bool
        Normalise the coils by their root sum of squares.
    toeplitz : bool
        Off the grid, use a point-spread convolution for the normal operator.

    Notes
    -----
    The step is the library's, and so is what it costs.  Each one runs a
    conjugate-gradient solve whose backward pass is another, by
    ``norm_inv``'s implicit differentiation rather than by unrolling --
    ``norm_inv_der_src`` and ``norm_inv_adj_src`` in ``nlops/norm_inv.c``.
    So the memory of a K-step unroll is K cells and not K times the inner
    iterations, and BART's ``nlop_checkpoint_create_F`` takes it down further.

    The coil weighting is worth knowing about before reading a gradient.  BART
    weights the coil half of the state by ``(1 + a |k|^2)^(-b/2)``, and its
    default ``b = 32`` is a sixteenth power: over the state of a small fit the
    gradient of that half spans tens of decades and its tail runs below
    float32's smallest normal number.  Below that edge the arithmetic is the
    platform's business rather than the library's -- a right-hand side whose
    norm is no longer a normal number is one BART's ``checkeps`` declines to
    iterate on, and the solve comes back untouched, with ``Warning: data
    corrupted`` in the log and a gradient of zeros.  Forward, none of this
    matters and the default is what ``nlinv`` reconstructs with; a *gradient*
    that has to be meaningful in the coil coefficients wants a gentler
    weighting, which is what ``sobolev=(220.0, 8.0)`` is.
    """

    def __init__(
        self,
        image_shape: Shape,
        pattern: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        *,
        iterations: int = 1,
        redu: float = 2.0,
        alpha_min: float = 0.0,
        batch: int = 1,
        cg_maxiter: int = 30,
        cg_tol: float = 0.0,
        cg_lambda: float = 0.0,
        kspace_shape: Shape | None = None,
        coil_shape: Shape | None = None,
        weights: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sobolev: tuple[float, float] = (220.0, 32.0),
        c: float = 1.0,
        real: bool = False,
        sos: bool = False,
        toeplitz: bool = True,
    ):
        self.image_shape = tuple(image_shape)
        coils, spatial = _spatial(self.image_shape)
        self.coil_image_shape = (coils, *spatial)

        self.noncart = trajectory is not None
        if pattern is None and not self.noncart:
            # Only the shape is kept: the pattern itself is an argument of
            # `prepare`, because a network is handed one per call.
            pattern = torch.ones((1, *spatial), dtype=torch.complex64)
        self.trajectory = _tensor(trajectory, "trajectory")
        self.pattern = _tensor(pattern, "pattern")
        self.weights = _tensor(weights, "weights")
        self.basis = _tensor(basis, "basis")
        self.mask = _tensor(mask, "mask")

        self.coil_shape = tuple(coil_shape) if coil_shape is not None else self.coil_image_shape
        if kspace_shape is not None:
            self.kspace_shape = tuple(kspace_shape)
        elif self.noncart:
            from bartorch.linop.nufft import default_kspace_shape

            self.kspace_shape = default_kspace_shape(
                tuple(self.trajectory.shape), self.coil_image_shape, 3
            )
        else:
            self.kspace_shape = self.coil_image_shape

        if 1 > int(iterations):
            raise ValueError("a Gauss-Newton operator takes at least one step")
        if 1 > int(batch):
            raise ValueError("a batch is at least one")

        self.iterations = int(iterations)
        self.redu = float(redu)
        self.alpha_min = float(alpha_min)
        self.batch = int(batch)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_tol = float(cg_tol)
        self.cg_lambda = float(cg_lambda)
        self.sobolev = (float(sobolev[0]), float(sobolev[1]))
        self.c = float(c)
        self.real = bool(real)
        self.sos = bool(sos)
        self.toeplitz = bool(toeplitz)

        self._model = self._make_model()
        # BART's model is built without a sampling pattern; `prepare` is what
        # gives it one, as a side effect of the gridding.  Applying a step
        # before that reads a diagonal nothing has written.
        self._prepared = False
        super().__init__()

    # --- the model ---------------------------------------------------------

    def _image_only(self) -> Shape:
        return (1,) + self.coil_image_shape[1:]

    def _make_model(self) -> _Handle:
        _ensure_ready()
        lib = library()
        trj_dims, _ = _optional(self.trajectory)
        # BART calls the sampling pattern the model's weights, and off the
        # grid that is what it is: the density compensation.
        wgh_dims, _ = _optional(self.weights if self.noncart else self.pattern)
        bas_dims, basis = _optional(self.basis)
        msk_dims, mask = _optional(self.mask)
        a, b = self.sobolev

        device = None
        for t in (self.trajectory, self.pattern, self.weights, self.mask):
            if t is not None:
                device = t.device
                break
        self.device = device

        with _lock, _on_device(device or torch.device("cpu")):
            ptr = lib.bartorch_noir_net_create(
                DIMS,
                dims(self.kspace_shape),
                dims(self.coil_image_shape),
                dims(self._image_only()),
                dims(self.coil_shape),
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
                int(self.real),
                int(self.sos),
                a,
                b,
                self.c,
                int(self.toeplitz),
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
        """The iterate ``nlinv`` starts from: an image of ones, and no coils.

        ``noir2_init`` writes exactly this.
        """
        made = torch.zeros(self.state_shape, dtype=torch.complex64, device=device)
        made[..., : math.prod(self._image_only())] = 1.0
        if batch is not None and batch != made.shape[0]:
            made = made[:1].expand(batch, *made.shape[1:]).contiguous()
        return made

    def weight(self, alpha: float, device=None) -> torch.Tensor:
        """``alpha`` as the operator takes it.

        BART's step multiplies the iterate by the weight rather than scaling
        it, so the weight is a vector as long as the iterate; this is that
        vector, and :meth:`forward` makes it for a caller who passes a number.
        """
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
        """``(kspace, pattern)`` -> the data a step takes.

        On the grid this is the adjoint transform with the pattern applied;
        off it, the adjoint NUFFT, and the pattern is the sampling weights.
        """
        return _Companion(self, library().bartorch_noir_net_adjoint, "prepare")

    def decompose(self) -> NonlinearOperator:
        """``x`` -> ``(image, sensitivities)``, through the model's transforms.

        The Sobolev weighting is in there, so what comes back is coil
        *profiles* and not the coefficients that were fitted.
        """
        return _Companion(self, library().bartorch_noir_net_decompose, "decompose")

    def split(self) -> NonlinearOperator:
        """``x`` -> ``(image, coefficients)``, with no transform applied."""
        return _Companion(self, library().bartorch_noir_net_split, "split")

    def join(self) -> NonlinearOperator:
        """``(image, coefficients)`` -> ``x``: :meth:`split` the other way."""
        return _Companion(self, library().bartorch_noir_net_join, "join")

    def __repr__(self) -> str:
        return (
            f"GaussNewton({self.image_shape}, iterations={self.iterations}, "
            f"batch={self.batch}, cg_maxiter={self.cg_maxiter})"
        )
