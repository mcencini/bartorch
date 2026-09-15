"""One step of each of BART's proximal iterations, as a torch module.

A block is the only copy of its iteration's step: a solver loops over one, and
a network stacks several.  ``state = block.start(y, A, x0)`` sets a run up,
``state = block(state, A)`` takes a step, and ``block.output(state, A)`` is the
image.  Step sizes and weights are parameters, frozen until ``requires_grad_()``;
while frozen, a step answers with the library's bits.  ``cclambda`` and
``precond`` are ``lsqr2_create``'s: the step sees ``M (A^H A + cclambda) x`` and
``M A^H y``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from torch import nn

from bartorch._operator import as_operand
from bartorch.priors.base import _as_terms

__all__ = ["ADMMBlock", "FISTABlock", "ISTBlock", "PRIDUBlock"]


# --- scalars ---------------------------------------------------------------------


def _setting(value: float) -> nn.Parameter:
    """``value`` as a frozen parameter, held in double so a frozen step reads it back exactly."""
    return nn.Parameter(torch.tensor(float(value), dtype=torch.float64), requires_grad=False)


def _value(p: nn.Parameter):
    """A setting as a step reads it: its float while frozen, a float32 tensor once learned."""
    return p.float() if p.requires_grad else float(p)


def _plain(v) -> float:
    """``v`` as a detached float: what steers the schedule, not the model."""
    return float(v.detach()) if isinstance(v, torch.Tensor) else float(v)


def _single(value):
    """``value`` as the single-precision number BART would have held.

    Every scalar in these iterations is a C ``float`` unless BART declares it a
    ``double``, so each is rounded where the library rounds it.  A learned
    tensor passes through: it is single precision already, and ``float()``
    would drop its graph.
    """
    if isinstance(value, torch.Tensor):
        return value
    return float(np.float32(value))


def _over(value, divisor: float):
    """``conf->step / maxeigen``: a float divided in double and rounded back."""
    if isinstance(value, torch.Tensor):
        return value / divisor
    return _single(_single(value) / divisor)


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """``vecops.c``'s ``dot``: products in single precision, summed in double, detached."""
    x = torch.view_as_real(a.detach()) if a.is_complex() else a.detach()
    z = torch.view_as_real(b.detach()) if b.is_complex() else b.detach()
    return float((x * z).double().sum())


def _norm(x: torch.Tensor) -> float:
    """``vecops.c``'s ``norm``: squares in single precision, summed and rooted in double.

    Detached: the norms steer step adaptation and stopping, which are the
    schedule rather than the model.
    """
    x = x.detach()
    parts = torch.view_as_real(x) if x.is_complex() else x
    return float(torch.sqrt((parts * parts).double().sum()))


def _ravine(told: float, t: float) -> tuple[float, float]:
    """``ravine``'s two coefficients, each operation in single precision."""
    one = np.float32(1.0)
    t32, told32 = np.float32(t), np.float32(told)
    return float((one - told32) / t32 - one), float((told32 - one) / t32 + one)


def _formula(p: float, q: float, r: float, t: float) -> float:
    """``fista_formula``, ``(p + sqrtf(q + r t^2)) / 2``, with the two roundings x86-64 makes.

    clang fuses ``q + r t t`` where the hardware has a fused multiply-add, which
    arm64 does; with BART's ``r = 4`` both readings agree at every step.
    """
    t32 = np.float32(t)
    inner = np.float32(np.float32(q) + np.float32(r) * t32 * t32)
    return float((np.float32(p) + np.sqrt(inner)) / np.float32(2.0))


def _halved(tau, k: int):
    """``hogwild``'s step at iteration ``k``: halved after 10 steps, then 20, then 40."""
    seen, period = 0, 10
    for _ in range(k + 1):
        seen += 1
        if seen == period:
            seen, period, tau = 0, period * 2, _single(tau / 2)
    return tau


def _empty(adjoint: torch.Tensor) -> bool:
    """``checkeps``: an adjoint with no norm, or one that is not a normal number."""
    eps = float(np.float32(float(torch.linalg.vector_norm(adjoint.detach()))))
    if 0.0 == eps:
        return True
    return not (math.isfinite(eps) and abs(eps) >= float(np.finfo(np.float32).tiny))


# --- operands ---------------------------------------------------------------------


def _batched(apply, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """``apply`` item by item over a leading batch axis, when ``x`` has one.

    BART builds an operator per shape and a batch is not one of its axes.
    """
    if x.ndim == len(shape) + 1:
        return torch.stack([apply(item) for item in x])
    return apply(x)


def _begin(y, A, x0):
    """The data as complex64, and where a run starts: ``x0``, or zero, where BART starts."""
    y = torch.as_tensor(y)
    y = _batched(lambda v: as_operand(v, A.oshape, "y"), y, A.oshape)
    if x0 is not None:
        x = _batched(lambda v: as_operand(v, A.ishape, "x0"), torch.as_tensor(x0), A.ishape)
        return y, x.clone()
    batch = y.shape[:1] if y.ndim == len(A.oshape) + 1 else ()
    return y, torch.zeros((*batch, *A.ishape), dtype=torch.complex64, device=y.device)


def _preconditioner(precond, shape):
    """``precond``, checked to map the image to itself."""
    if precond is not None and not (tuple(precond.ishape) == tuple(precond.oshape) == tuple(shape)):
        raise ValueError(
            f"a preconditioner maps the image to itself, so it is {tuple(shape)} to "
            f"{tuple(shape)}, not {tuple(precond.ishape)} to {tuple(precond.oshape)}"
        )
    return precond


def _adjoint(A, y: torch.Tensor, precond=None) -> torch.Tensor:
    """``M A^H y``, recorded when ``y`` is."""
    from bartorch.linop.autograd import apply_adjoint
    from bartorch.linop.base import _tracking

    one = (lambda v: apply_adjoint(A, v)) if _tracking(y) else A.adjoint
    out = _batched(one, y, A.oshape)
    return out if precond is None else _batched(precond, out, A.ishape)


def _normal(A, x: torch.Tensor, cclambda: float, precond=None) -> torch.Tensor:
    """``M (A^H A x + cclambda x)``, in ``normaleq_l2_apply``'s order; recorded when ``x`` is."""
    from bartorch.linop.autograd import apply_normal
    from bartorch.linop.base import _tracking

    one = (lambda v: apply_normal(A, v)) if _tracking(x) else A.normal
    out = _batched(one, x, A.ishape)
    out = out + cclambda * x if cclambda else out
    return out if precond is None else _batched(precond, out, A.ishape)


def _prox(term, w: torch.Tensor, gamma, image_shape) -> torch.Tensor:
    if getattr(term, "_batches", False):
        return term.prox(w, gamma, image_shape=image_shape)
    return _batched(
        lambda v: term.prox(v, gamma, image_shape=image_shape), w, term.prox_shape(image_shape)
    )


def _transform(term, x: torch.Tensor, image_shape, mode: str = "forward") -> torch.Tensor:
    if getattr(term, "_batches", False):
        return term.apply_transform(x, image_shape, mode=mode)
    shape = term.prox_shape(image_shape) if "adjoint" == mode else image_shape
    return _batched(lambda v: term.apply_transform(v, image_shape, mode=mode), x, shape)


def _terms(priors, name: str) -> list:
    terms = _as_terms(priors)
    if any(getattr(t, "_extends", False) for t in terms):
        raise TypeError(
            f"{name} thresholds the image, and a term that adds unknowns walks the fields "
            "behind it too; ADMMBlock and PRIDUBlock take it"
        )
    return terms


def _refuse_preconditioner(precond, name: str) -> None:
    if precond is not None:
        raise ValueError(
            f"{name} takes no preconditioner with a term that adds unknowns: it maps the "
            "image, and the step walks the image and the fields behind it"
        )


@dataclasses.dataclass(frozen=True)
class _Space:
    """The vector a step walks with a term that adds unknowns: the image, then those unknowns."""

    terms: list
    A: object
    image: tuple[int, ...]


def _space(terms, A, precond, name: str) -> _Space | None:
    """The set's penalties and the encoding over the longer vector, as ``pics.c`` chains it."""
    if not any(getattr(t, "_extends", False) for t in terms):
        return None
    from bartorch.linop.shape import Extract, Reshape
    from bartorch.optim.linear import _penalties

    _refuse_preconditioner(precond, name)
    total = math.prod(A.ishape)
    penalties, svars = _penalties(terms, tuple(A.ishape))
    encoding = A @ Reshape(A.ishape, (total,)) @ Extract((0,), (total,), (total + svars,))
    return _Space(penalties, encoding, tuple(A.ishape))


def _walked(terms, state, A):
    """The terms and encoding a step walks: its own, or those ``start`` set up."""
    space = state.space
    return (terms, A) if space is None else (space.terms, space.A)


def _lengthen(x: torch.Tensor, image: tuple[int, ...], shape: tuple[int, ...]) -> torch.Tensor:
    """The image in front of zeros for the unknowns behind it, where ``pics`` starts them."""
    batch = x.shape[: x.ndim - len(image)]
    front = x.reshape(*batch, -1)
    behind = torch.zeros((*batch, shape[0] - front.shape[-1]), dtype=x.dtype, device=x.device)
    return torch.cat([front, behind], -1)


def _front(state, A) -> torch.Tensor:
    """The image: the state's ``x``, or the front of the longer vector."""
    if state.space is None:
        return state.x
    batch = state.x.shape[:-1]
    return state.x[..., : math.prod(A.ishape)].reshape(*batch, *A.ishape)


# --- iterative soft thresholding ---------------------------------------------------


class ISTBlock(nn.Module):
    """One step of iterative soft thresholding, ``italgos.c``'s ``ist``.

    Thresholds, then steps along ``A^H y - (A^H A + cclambda) x``;
    :meth:`output` thresholds once more, as BART does after its loop.  With
    ``eigen`` the step is divided by the largest eigenvalue, estimated at
    :meth:`start`.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        x: torch.Tensor
        adjoint: torch.Tensor
        divisor: float = 1.0
        k: int = 0

    def __init__(
        self,
        prior,
        *,
        step: float = 0.95,
        eigen: bool = False,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__()
        terms = _terms(prior, type(self).__name__)
        if 1 != len(terms):
            raise ValueError(f"{type(self).__name__} takes exactly one term, not {len(terms)}")
        self.prior = terms[0]
        self.step = _setting(step)
        self.eigen = bool(eigen)
        self.cclambda = float(cclambda)
        self.precond = precond

    def start(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> State:
        """The run's state: the start, and ``A^H y`` kept for every step."""
        from bartorch.optim.linear import maxeigen

        y, x = _begin(y, A, x0)
        self.prior.rewind(A.ishape)
        adjoint = _adjoint(A, y, _preconditioner(self.precond, A.ishape))
        divisor = 1.0
        if self.eigen and not _empty(adjoint):
            divisor = maxeigen(A, cclambda=self.cclambda, precond=self.precond)
        return self.State(x, adjoint, divisor)

    def _tau(self, state: State, k: int):
        return _over(_value(self.step), state.divisor)

    def forward(self, state: State, A) -> State:
        tau = self._tau(state, state.k)
        x = _prox(self.prior, state.x, tau, A.ishape)
        x = x - tau * (_normal(A, x, self.cclambda, self.precond) - state.adjoint)
        return dataclasses.replace(state, x=x, k=state.k + 1)

    def output(self, state: State, A) -> torch.Tensor:
        """The image, thresholded once more, as ``italgo_config`` leaves ``last`` false."""
        return _prox(self.prior, state.x, self._tau(state, max(state.k - 1, 0)), A.ishape)


class FISTABlock(ISTBlock):
    """One step of fast iterative soft thresholding, ``italgos.c``'s ``fista``.

    :class:`ISTBlock` with Nesterov's ravine step between the threshold and the
    gradient; the momentum is ``fista_formula`` with ``pqr``.  ``hogwild``
    halves the step after 10 steps, then 20, then 40.
    """

    @dataclasses.dataclass(frozen=True)
    class State(ISTBlock.State):
        previous: torch.Tensor | None = None
        t: float = 1.0

    def __init__(
        self,
        prior,
        *,
        step: float = 0.95,
        eigen: bool = False,
        hogwild: bool = False,
        pqr: tuple[float, float, float] | None = None,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(prior, step=step, eigen=eigen, cclambda=cclambda, precond=precond)
        self.hogwild = bool(hogwild)
        self.pqr = (1.0, 1.0, 4.0) if pqr is None else tuple(float(v) for v in pqr)

    def start(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> State:
        state = super().start(y, A, x0)
        return dataclasses.replace(state, previous=state.x)

    def _tau(self, state: State, k: int):
        tau = super()._tau(state, k)
        return _halved(tau, k) if self.hogwild else tau

    def forward(self, state: State, A) -> State:
        tau = self._tau(state, state.k)
        z = _prox(self.prior, state.x, tau, A.ishape)

        # Two axpys rather than the combination they add up to: `x + c x` and
        # `(1 + c) x` are not the same float32 number.
        t = _formula(*self.pqr, state.t)
        before, after = _ravine(state.t, t)
        x = state.previous
        x = x + before * x
        x = x + after * z

        x = x - tau * (_normal(A, x, self.cclambda, self.precond) - state.adjoint)
        return dataclasses.replace(state, x=x, previous=z, t=t, k=state.k + 1)


# --- alternating directions --------------------------------------------------------


class _Resolvent(torch.autograd.Function):
    """``x = K^-1 b`` with ``K = N + rho S``; the backward pass is one more solve with ``K``.

    ``dx = K^-1 (db - drho S x)``, so ``b`` gets ``w = K^-1 g`` and ``rho`` gets
    ``-Re <w, S x>``.  The warm start carries no gradient.
    """

    @staticmethod
    def forward(ctx, b, rho, solve, spread):  # noqa: D102
        ctx.solve, ctx.spread = solve, spread
        with torch.no_grad():
            x = solve(b, True)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        (x,) = ctx.saved_tensors
        w = ctx.solve(g.resolve_conj().contiguous(), False)
        rho = None
        if ctx.needs_input_grad[1]:
            rho = -torch.real(torch.sum(w.conj() * ctx.spread(x))).float()
        return w, rho, None, None


class ADMMBlock(nn.Module):
    """One step of the alternating direction method of multipliers, ``admm.c``'s ``admm``.

    For ``min 0.5 ||A x - y||^2 + sum_j f_j(G_j x - b_j)``: conjugate gradients on
    ``A^H A + cclambda + rho sum_j G_j^H G_j`` from the previous ``x``, then each
    term's split and dual.  The state's ``done`` is Boyd's residual test, and
    ``invokes`` counts inner iterations, which is what BART's ``maxiter`` budgets.
    Each step takes its own ``rho`` unless ``dynamic_rho`` or ``hogwild`` moves
    it, and then the state carries it.  With a term that adds unknowns the state
    walks the image followed by them, and :meth:`output` is the image.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        x: torch.Tensor
        adjoint: torch.Tensor
        z: tuple
        u: tuple
        rho: float | torch.Tensor
        tau: float = 2.0
        k: int = 0
        invokes: int = 0
        hogwild: tuple[int, int] = (0, 1)
        done: bool = False
        space: _Space | None = None

    #: `cg_xupdate`'s tolerance, relative to the right-hand side.
    _cg_eps = 1e-3

    def __init__(
        self,
        priors,
        *,
        rho: float = 0.5,
        alpha: float = 1.6,
        cg_maxiter: int = 10,
        cg_maxiter_first: int | None = None,
        cclambda: float = 0.0,
        biases=None,
        dynamic_rho: bool = False,
        dynamic_tau: bool = False,
        relative_norm: bool = False,
        fast: bool = False,
        hogwild: bool = False,
        mu: float = 3.0,
        tau_max: float = 20.0,
        abstol: float = 0.0,
        reltol: float = 0.0,
        precond=None,
    ):
        super().__init__()
        self.terms = _as_terms(priors)
        self.learned = nn.ModuleList([t for t in self.terms if isinstance(t, nn.Module)])
        self.biases = [None] * len(self.terms) if biases is None else list(biases)
        if len(self.biases) != len(self.terms):
            raise ValueError("one bias per term, or none at all")
        if hogwild and dynamic_rho:
            raise ValueError("BART's ADMM takes hogwild or a dynamic rho, not both")
        if fast and dynamic_rho:
            raise ValueError("a dynamic rho needs the residuals, which fast mode does not compute")

        self.rho = _setting(rho)
        self.alpha = _setting(alpha)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_maxiter_first = None if cg_maxiter_first is None else int(cg_maxiter_first)
        self.cclambda = float(cclambda)
        self.dynamic_rho = bool(dynamic_rho)
        self.dynamic_tau = bool(dynamic_tau)
        self.relative_norm = bool(relative_norm)
        self.fast = bool(fast)
        self.hogwild = bool(hogwild)
        self.mu = float(mu)
        self.tau_max = float(tau_max)
        self.abstol = float(abstol)
        self.reltol = float(reltol)
        self.precond = precond

    def start(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> State:
        """The run's state: the start, zero splits and duals, and ``A^H y``."""
        y, x = _begin(y, A, x0)
        space = _space(self.terms, A, self.precond, type(self).__name__)
        if space is not None and any(b is not None for b in self.biases):
            raise ValueError("a set holding a term that adds unknowns takes no biases")
        terms, walked = (self.terms, A) if space is None else (space.terms, space.A)
        if space is not None:
            x = _lengthen(x, A.ishape, walked.ishape)
        for term in terms:
            term.rewind(walked.ishape)
        batch = x.shape[: x.ndim - len(walked.ishape)]
        z = tuple(
            torch.zeros((*batch, *t.prox_shape(walked.ishape)), dtype=x.dtype, device=x.device)
            for t in terms
        )
        u = tuple(torch.zeros_like(zj) for zj in z)
        adjoint = _adjoint(walked, y, _preconditioner(self.precond, A.ishape))
        return self.State(x, adjoint, z, u, _single(_value(self.rho)), space=space)

    def forward(self, state: State, A) -> State:
        terms, A = _walked(self.terms, state, A)
        biases = self.biases if state.space is None else [None] * len(terms)
        shape = A.ishape
        moves = self.dynamic_rho or self.hogwild
        rho = state.rho if moves else _single(_value(self.rho))
        tau = state.tau
        alpha = _value(self.alpha)
        z, u = list(state.z), list(state.u)

        rhs = torch.zeros_like(state.x)
        for j, term in enumerate(terms):
            r = z[j] - u[j]
            if biases[j] is not None:
                r = r + biases[j]
            rhs = rhs + _transform(term, r, shape, "adjoint")
        rhs = rho * rhs + state.adjoint

        x, spent = self._solve_x(terms, state.x, rhs, rho, A, 0 == state.k)

        n1 = n2 = r_sq = 0.0
        s = torch.zeros_like(x)
        gh_usum = torch.zeros_like(x)

        for j, term in enumerate(terms):
            bias = biases[j]
            gx = _transform(term, x, shape)
            z_old = z[j]

            if not self.fast:
                residual = gx
                n1 += _norm(residual) ** 2
                gx = alpha * gx + (1.0 - alpha) * z[j]
                if bias is not None:
                    gx = gx + (1.0 - alpha) * bias

            w = gx + u[j]
            if bias is not None:
                w = w - bias

            z[j] = _prox(term, w, 1.0 / rho, shape) if rho else w
            u[j] = w - z[j]

            if not self.fast:
                r = residual - z[j]
                if bias is not None:
                    r = r - bias
                # `float r_norm` against `double n1, n2`, as `admm.c` declares them.
                r_sq = _single(r_sq + _norm(r) ** 2)
                s = s + _transform(term, z[j] - z_old, shape, "adjoint")
                gh_usum = gh_usum + _transform(term, u[j], shape, "adjoint")
                n2 += _norm(z[j]) ** 2

        done = False
        hogwild = state.hogwild
        if not self.fast:
            weight = _plain(rho)
            r_norm = _single(math.sqrt(r_sq))
            s_norm = _single(weight * _norm(s))
            n3 = sum(_norm(b) ** 2 for b in biases if b is not None)
            r_scaling = math.sqrt(max(n1, n2, n3))
            s_scaling = weight * _norm(gh_usum)

            # BART counts real numbers, twice the complex ones.
            m = 2 * sum(math.prod(t.prox_shape(shape)) for t in terms)
            n = 2 * math.prod(tuple(x.shape))
            eps_pri = _single(self.abstol * math.sqrt(m) + self.reltol * r_scaling)
            eps_dual = _single(self.abstol * math.sqrt(n) + self.reltol * s_scaling)

            done = r_norm < eps_pri and s_norm < eps_dual
            rho, tau, u, hogwild = self._adapt(
                rho, tau, r_norm, s_norm, r_scaling, s_scaling, u, hogwild
            )
        else:
            rho, tau, u, hogwild = self._adapt(rho, tau, 0.0, 0.0, 1.0, 1.0, u, hogwild)

        return dataclasses.replace(
            state,
            x=x,
            z=tuple(z),
            u=tuple(u),
            rho=rho,
            tau=tau,
            k=state.k + 1,
            invokes=state.invokes + spent,
            hogwild=hogwild,
            done=done,
        )

    def output(self, state: State, A) -> torch.Tensor:
        return _front(state, A)

    def _solve_x(self, terms, x, rhs, rho, A, first: bool):
        """``cg_xupdate``: conjugate gradients from ``x``, and the iterations it took.

        A batch is solved item by item, so no item steers another's stopping;
        the count is the worst item's.
        """
        if x.ndim == len(A.ishape) + 1:
            made, worst = [], 0
            for item, side in zip(x, rhs):
                out, spent = self._solve_x(terms, item, side, rho, A, first)
                made.append(out)
                worst = max(worst, spent)
            return torch.stack(made), worst

        from bartorch.linop import Identity
        from bartorch.linop.base import LinearOperator, _tracking, _WithNormal
        from bartorch.optim.linear import CG

        if 0.0 == float(torch.linalg.vector_norm(rhs.detach())):
            return x, 0

        shape, weight = A.ishape, _plain(rho)

        def spread(v):
            out = None
            for term in terms:
                part = term.apply_transform(v, shape, mode="normal")
                out = part if out is None else out + part
            return out

        def apply(v):
            return self._xupdate_normal(A, weight, v, terms=terms)

        def transpose(v):
            return self._xupdate_normal(A, weight, v, transposed=True, terms=terms)

        budget = self.cg_maxiter
        if first and self.cg_maxiter_first is not None:
            budget = self.cg_maxiter_first
        operator = _WithNormal(
            Identity(shape), LinearOperator.from_callbacks(shape, shape, apply, apply, apply)
        )
        steps: list[int] = []

        def solve(b, warm):
            solver = CG(maxiter=budget, tol=self._cg_eps)
            if warm:
                return solver(b, operator, x0=x.detach(), steps=steps)
            if self.precond is None:
                return solver(b, operator)
            transposed = LinearOperator.from_callbacks(
                shape, shape, transpose, transpose, transpose
            )
            return solver(b, _WithNormal(Identity(shape), transposed))

        learned = isinstance(rho, torch.Tensor) and rho.requires_grad
        if _tracking(rhs) or learned:
            rate = rho if isinstance(rho, torch.Tensor) else torch.tensor(rho)
            out = _Resolvent.apply(rhs, rate, solve, spread)
        else:
            out = solve(rhs, True)
        return out, steps[0]

    def _xupdate_normal(self, A, rho: float, v: torch.Tensor, transposed: bool = False, terms=None):
        """``K v`` for ``K = rho sum_j G_j^H G_j + M (A^H A + cclambda)``, or ``K^H v``.

        ``admm_normaleq``'s order: the terms summed first, each scaled by
        ``rho``, and the data term added last.
        """
        out = None
        for term in self.terms if terms is None else terms:
            part = rho * term.apply_transform(v, A.ishape, mode="normal")
            out = part if out is None else out + part
        precond = self.precond
        w = precond.adjoint(v) if transposed and precond is not None else v
        normal = A.normal(w)
        if self.cclambda:
            normal = normal + self.cclambda * w
        if not transposed and precond is not None:
            normal = precond.forward(normal)
        return normal if out is None else out + normal

    def _adapt(self, rho, tau, r_norm, s_norm, r_scaling, s_scaling, u, hogwild):
        """BART's ``tau`` and ``rho`` moves, and hogwild's doubling."""
        sc = 1.0

        if self.dynamic_tau:
            # `sqrt(r_norm / s_norm)` over two floats; with both zero it is a
            # NaN, every comparison is false and `tau` goes to its ceiling.
            t = math.sqrt(_single(r_norm / s_norm)) if s_norm else float("nan")
            if self.tau_max > t >= 1.0:
                tau = _single(t)
            elif 1.0 > t > 1.0 / self.tau_max:
                tau = _single(1.0 / t)
            else:
                tau = _single(self.tau_max)

        if self.dynamic_rho:
            r, s = r_norm, s_norm
            if self.relative_norm:
                r, s = r / r_scaling, s / s_scaling
            if r > self.mu * s:
                sc = tau
            elif s > self.mu * r:
                sc = _single(1.0 / tau)

        if self.hogwild:
            k, K = hogwild[0] + 1, hogwild[1]
            if k == K:
                k, K, sc = 0, K * 2, 2.0
            hogwild = (k, K)

        if 1.0 != sc:
            rho = _single(rho * sc)
            # `smul(z_dims[j], 1. / sc, u[j], u[j])`: the reciprocal once, as a float.
            back = _single(1.0 / sc)
            u = [back * uj for uj in u]

        return rho, tau, u, hogwild


# --- primal and dual ----------------------------------------------------------------


class PRIDUBlock(nn.Module):
    """One step of BART's primal-dual iteration, ``italgos.c``'s ``chambolle_pock``.

    The data term is its own dual, updated through its resolvent, and so is
    every term but the first when that one's transform is the identity: it
    becomes the primal step.  ``sigma`` and ``tau`` are ``sqrt(step)`` times and
    over ``sigma_tau_ratio``, divided with ``eigen`` by the root of the largest
    eigenvalue; ``hogwild`` decays by 0.95 a step.  Each step takes its own
    ``sigma`` and ``tau`` unless ``adaptive_step`` moves them, and then the state
    carries them.  ``done`` is BART's absolute tolerance on the two residuals.
    With a term that adds unknowns the state walks the image followed by them,
    and :meth:`output` is the image.
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        x: torch.Tensor
        adjoint: torch.Tensor
        avg: torch.Tensor
        adjoint_dual: torch.Tensor
        duals: tuple
        sigma: float | torch.Tensor
        tau: float | torch.Tensor
        primal: bool
        divisor: float = 1.0
        k: int = 0
        done: bool = False
        space: _Space | None = None

    def __init__(
        self,
        priors,
        *,
        step: float = 0.95,
        sigma_tau_ratio: float = 1.0,
        adaptive_step: bool = False,
        eigen: bool = False,
        hogwild: bool = False,
        cclambda: float = 0.0,
        tol: float = 1e-4,
        precond=None,
    ):
        super().__init__()
        self.terms = _as_terms(priors)
        self.learned = nn.ModuleList([t for t in self.terms if isinstance(t, nn.Module)])
        root = float(np.float32(math.sqrt(step)))
        self.ratio = float(np.float32(sigma_tau_ratio))
        self.sigma = _setting(np.float32(root * self.ratio))
        self.tau = _setting(np.float32(root / self.ratio))
        self.adaptive_step = bool(adaptive_step)
        self.eigen = bool(eigen)
        self.hogwild = bool(hogwild)
        self.cclambda = float(cclambda)
        self.tol = float(tol)
        self.precond = precond

    def start(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> State:
        """The run's state: the split ``iter2_chambolle_pock`` makes, zero duals, and the steps."""
        from bartorch.optim.linear import maxeigen

        y, x = _begin(y, A, x0)
        space = _space(self.terms, A, self.precond, type(self).__name__)
        every, walked = (self.terms, A) if space is None else (space.terms, space.A)
        if space is not None:
            x = _lengthen(x, A.ishape, walked.ishape)
        for term in every:
            term.rewind(walked.ishape)
        primal = bool(every) and every[0].transform_is_identity(walked.ishape)
        duals = every[1:] if primal else every

        # Estimated over the encoding and the dual terms' transforms together.
        divisor = 1.0
        if self.eigen:
            divisor = math.sqrt(
                maxeigen(walked, duals, cclambda=self.cclambda, precond=self.precond)
            )
        batch = x.shape[: x.ndim - len(walked.ishape)]
        return self.State(
            x=x,
            adjoint=_adjoint(walked, y, _preconditioner(self.precond, A.ishape)),
            avg=x,
            adjoint_dual=torch.zeros_like(x),
            duals=tuple(
                torch.zeros((*batch, *t.prox_shape(walked.ishape)), dtype=x.dtype, device=x.device)
                for t in duals
            ),
            sigma=_over(_value(self.sigma), divisor),
            tau=_over(_value(self.tau), divisor),
            primal=primal,
            divisor=divisor,
            space=space,
        )

    def forward(self, state: State, A) -> State:
        every, A = _walked(self.terms, state, A)
        shape = A.ishape
        primal = every[0] if state.primal else None
        terms = every[1:] if state.primal else every
        x, avg, adjoint_dual, k = state.x, state.avg, state.adjoint_dual, state.k
        duals = list(state.duals)
        sigma, tau = state.sigma, state.tau
        if not self.adaptive_step:
            sigma = _over(_value(self.sigma), state.divisor)
            tau = _over(_value(self.tau), state.divisor)
        # `float lambda = (float)pow(decay, i)`, from a float32 `decay`.
        lam = float(np.float32(float(np.float32(0.95)) ** k)) if self.hogwild else 1.0

        # The data term's dual, through its resolvent: both coefficients are
        # worked out in double and rounded to the float each vector is scaled by.
        previous = adjoint_dual
        step = sigma * _normal(A, avg, self.cclambda, self.precond) + adjoint_dual
        keep = _single(1.0 / (1.0 + sigma))
        pull = _single(-1.0 * sigma / (1.0 + sigma))
        fresh = keep * step + pull * state.adjoint
        adjoint_dual = lam * fresh + (1.0 - lam) * previous
        change = (adjoint_dual - previous).detach()
        moved = float(torch.real((change.conj() * change).sum()))

        # Each term's, through the conjugate of its prox.
        for j, term in enumerate(terms):
            # `axpy(u_old, 1. / sigma, u[j])`: the reciprocal once, as a float.
            over = _transform(term, avg, shape) + _single(1.0 / sigma) * duals[j]
            thresholded = _prox(term, over, 1.0 / sigma, shape)
            fresh_j = sigma * over - sigma * thresholded
            was = duals[j]
            duals[j] = lam * fresh_j + (1.0 - lam) * was
            moved_j = (duals[j] - was).detach()
            moved += float(torch.real((moved_j.conj() * moved_j).sum()))

        # The primal step.
        previous_x = x
        x = x - tau * adjoint_dual
        for j, term in enumerate(terms):
            x = x - tau * _transform(term, duals[j], shape, "adjoint")
        stepped = x if primal is None else _prox(primal, x, tau, shape)
        x = lam * stepped + (1.0 - lam) * previous_x

        # `res2` is measured against `tau` before the adaptation.
        res2 = _single(_single(math.sqrt(max(moved, 0.0))) / _plain(tau))
        res1 = _single(_single(_norm(x - previous_x)) / _plain(sigma))

        if self.adaptive_step:
            sigma, tau = self._adapt((x - previous_x).detach(), sigma, tau, terms, A)

        avg = (1.0 + 1.0) * x - 1.0 * previous_x

        return dataclasses.replace(
            state,
            x=x,
            avg=avg,
            adjoint_dual=adjoint_dual,
            duals=tuple(duals),
            sigma=sigma,
            tau=tau,
            k=k + 1,
            # `iter2_chambolle_pock` leaves `eps` at one: the tolerance is absolute.
            done=self.tol > (res1 + res2),
        )

    def output(self, state: State, A) -> torch.Tensor:
        return _front(state, A)

    def _adapt(self, delta, sigma, tau, terms, A):
        """The move over what the operator makes of it, clipped under ``sqrt(sigma tau)``."""
        shape = A.ishape
        # `float norm_Kx`, each `+=` rounded back to a float.
        squared = 0.0
        for term in terms:
            squared = _single(squared + _norm(_transform(term, delta, shape)) ** 2)
        squared = _single(squared + _dot(_normal(A, delta, self.cclambda, self.precond), delta))

        norm_kx = _single(math.sqrt(max(squared, 0.0)))
        if 0.0 == norm_kx:
            return sigma, tau

        # Single-precision operations down to the literal: `0.95f` is not 0.95.
        ratio = _single(_single(_norm(delta)) / norm_kx)
        root = _single(math.sqrt(_single(_plain(sigma) * _plain(tau))))
        threshold = _single(_single(0.95) * root)
        chosen = min(threshold, ratio) if 0.0 != ratio < root else root
        return _single(chosen * self.ratio), _single(chosen / self.ratio)
