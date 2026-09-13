"""BART's proximal iterations, written as ``deepinv`` optimizers.

The iterations are BART's, step for step: what is here is the same arithmetic
in the same order, with every operator still the library's -- the normal
operator, and the proximal operator of each :mod:`bartorch.prox` term.  What
is left for torch is a handful of axpys on an image per step, which is nothing
beside a transform, and the tests hold each iteration against the library's own
to the bit.

What that buys is the shape.  A ``deepinv.optim.optim_iterators.OptimIterator``
goes into ``deepinv.optim.optim_builder``, and so into ``BaseOptim`` with
``unfold=True`` for an unrolled network or ``DEQ`` for a deep-equilibrium fixed
point -- neither of which an iteration running inside the library can be part
of, because there is nothing to differentiate through.

``deepinv`` is imported on first use rather than at import time -- it is a
dependency, but importing it is not free, and a script that only builds
operators should not pay for it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Built on first use, so that importing this module does not import deepinv;
# `__getattr__` below is what resolves them.
__all__ = [  # noqa: F822
    "ADMMIteration",
    "FISTAIteration",
    "ISTIteration",
    "PRIDUIteration",
    "NormalEquations",
    "TermPrior",
    "AsTerm",
]


def _batched(apply, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """``apply`` over a leading batch axis, when ``x`` has one more axis than ``shape``."""
    if x.ndim == len(shape) + 1:
        return torch.stack([apply(item) for item in x])
    return apply(x)


def _classes():
    """The ``deepinv`` bases, imported on first use."""
    try:
        from deepinv.optim import DataFidelity, Prior
        from deepinv.optim.optim_iterators import OptimIterator
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "bartorch.optim's iterations are written as deepinv optimizers, and "
            "deepinv is a dependency of this package -- an environment without it "
            "is a broken one rather than a lean one"
        ) from exc
    return DataFidelity, Prior, OptimIterator


def _built(name: str):
    """One of the classes below, built once, when ``deepinv`` is present."""
    if name not in _CACHE:
        _CACHE[name] = _BUILDERS[name]()
    return _CACHE[name]


_CACHE: dict[str, type] = {}


def _normal_equations() -> type:
    DataFidelity, _, _ = _classes()

    class NormalEquations(DataFidelity):
        """``||A x - y||^2 / 2``, differentiated through BART's normal operator.

        The gradient is ``A^H A x - A^H y``.  Written that way rather than as
        ``A^H (A x - y)`` it is one application of the operator's own normal,
        which for an encoding built with ``toeplitz=True`` is a convolution
        with a point spread function rather than a transform and its adjoint.
        That is the difference the whole encoding was built for, and it would
        be lost by taking the obvious route.

        ``A^H y`` does not change during a solve, so it is computed once and
        kept, keyed by the tensor it came from.  A physics that is not one of
        ours falls back to ``A_adjoint(A(x) - y)``.
        """

        def __init__(self, weight: float = 0.0):
            super().__init__()
            self.weight = float(weight)
            self._adjoint: tuple | None = None

        def _adjoint_data(self, y: torch.Tensor, physics) -> torch.Tensor:
            key = (id(y), y._version, y.shape)
            if self._adjoint is None or self._adjoint[0] != key:
                op = physics.op
                self._adjoint = (key, _batched(op.adjoint, y, op.oshape))
            return self._adjoint[1]

        def normal(self, x: torch.Tensor, physics) -> torch.Tensor:
            """``A^H A x + lambda x``, the operator ``lsqr`` builds.

            ``lambda`` is ``pics -q``, a weight on an identity added to the
            normal operator; ``normaleq_l2_apply`` applies the encoding's
            normal and then adds it, and so does this.
            """
            op = getattr(physics, "op", None)
            out = (
                _batched(op.normal, x, op.ishape)
                if op is not None
                else physics.A_adjoint(physics.A(x))
            )
            return out + self.weight * x if self.weight else out

        def grad(self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs):
            op = getattr(physics, "op", None)
            if op is None and not self.weight:
                return physics.A_adjoint(physics.A(x) - y)
            return self.normal(x, physics) - self._adjoint_data(y, physics)

        def fn(self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs):
            residual = physics.A(x) - y
            out = 0.5 * residual.flatten(1).abs().pow(2).sum(-1)
            if self.weight:
                out = out + 0.5 * self.weight * x.flatten(1).abs().pow(2).sum(-1)
            return out

    return NormalEquations


def _term_prior() -> type:
    _, Prior, _ = _classes()

    class TermPrior(Prior):
        """A :class:`bartorch.prox.Regularizer` as a ``deepinv`` prior.

        ``prox`` is BART's own proximal operator for the term, so an iteration
        driven by this prior calls what the library would have called.

        Parameters
        ----------
        term : Regularizer
            The term from :mod:`bartorch.prox`.
        image_shape : tuple of int
            What the term was configured for.  Needed because a tensor with a
            leading batch axis is applied item by item: BART builds a term's
            operator per shape, and a batch is not one of its axes.
        """

        def __init__(self, term, image_shape):
            super().__init__()
            self.term = term
            self.image_shape = tuple(image_shape)
            self.explicit_prior = False

        def prox(self, x: torch.Tensor, *args, gamma: float = 1.0, **kwargs) -> torch.Tensor:
            shape = self.term.prox_shape(self.image_shape)
            return _batched(
                lambda item: self.term.prox(item, gamma, image_shape=self.image_shape), x, shape
            )

        def __repr__(self) -> str:
            return f"TermPrior({self.term!r})"

    return TermPrior


def _as_term() -> type:
    _, Prior, _ = _classes()

    class AsTerm:
        """A ``deepinv`` prior or denoiser where a :mod:`bartorch.prox` term goes.

        The iterations here ask a term for four things: the transform in front
        of its proximal operator, the shape that transform lands in, the
        proximal operator itself, and a rewind of whatever randomness it
        carries.  A prior has none of the first, all of the third, and nothing
        to rewind -- so this answers *identity*, *the image's shape*, the
        prior's own ``prox``, and *nothing*.

        That is what lets a denoiser stand in a split or a dual: an
        alternating-direction solve with a plug-and-play prior is the same
        loop with this in place of a term.

        Parameters
        ----------
        prior : deepinv.optim.Prior
            Or anything with a ``prox(x, *args, gamma=...)``.  A bare
            denoiser is wrapped in ``deepinv.optim.PnP``; a complex image
            needs ``deepinv.models.to_complex_denoiser`` around it first,
            which this does not do for you, because whether a denoiser is
            complex-capable is not something to guess at.
        g_param : float, optional
            The prior's own parameter -- a denoiser's noise level, say --
            passed as ``deepinv`` passes it, before ``gamma``.
        """

        def __init__(self, prior, g_param: float | None = None, *, batched: bool = False):
            if not hasattr(prior, "prox"):
                if not callable(prior):
                    raise TypeError(
                        f"a regularizer is a term from bartorch.prox, a deepinv prior, "
                        f"or a denoiser -- not {prior!r}"
                    )
                from deepinv.optim import PnP

                prior = PnP(denoiser=prior)
            self.prior = prior
            self.g_param = g_param
            self.batched = bool(batched)

        # --- what a term answers ------------------------------------------

        def prox(self, x: torch.Tensor, gamma: float = 1.0, *, image_shape=None):
            # A denoiser is a network over a batch of images and an image here
            # is one image, so it is given an axis to be a batch of one on --
            # which for a three-axis image is the N of the N, C, H, W most of
            # deepinv's models expect.  A caller whose image already carries
            # that axis says so with `batched`.
            if self.batched:
                return self.prior.prox(x, self.g_param, gamma=gamma)
            return self.prior.prox(x[None], self.g_param, gamma=gamma)[0]

        def prox_shape(self, image_shape):
            return tuple(image_shape)

        def apply_transform(self, x: torch.Tensor, image_shape=None, mode: str = "forward"):
            return x

        def transform_is_identity(self, image_shape) -> bool:
            return True

        def rewind(self, image_shape) -> None:
            return None

        def __repr__(self) -> str:
            return f"AsTerm({self.prior!r})"

    return AsTerm


def _ist() -> type:
    _, _, OptimIterator = _classes()

    class ISTIteration(OptimIterator):
        r"""BART's iterative soft thresholding, one step of it.

        ``italgos.c``'s ``ist``, in order::

            x <- prox(x, tau)
            r <- A^H y - A^H A x
            x <- x + tau r

        with a last ``prox`` after the loop, because ``italgo_config`` leaves
        ``last`` false.  ``tau`` is ``alpha * step / maxeigen``; a ``hogwild``
        run halves it after 10 steps, then 20, then 40.

        The threshold comes first, which is not where a textbook proximal
        gradient step puts it.  It is where BART puts it, and the last ``prox``
        is what makes the two agree at the end.
        """

        def __init__(self, **kwargs):
            kwargs.setdefault("has_cost", False)
            super().__init__(**kwargs)

        def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *a, **kw):
            x = X["est"][0]
            k = X.get("it", 0)
            tau = _tau(cur_params, k)

            x = cur_prior.prox(x, cur_params.get("g_param"), gamma=_scale(cur_params, k) * tau)
            x = x - tau * cur_data_fidelity.grad(x, y, physics)

            return {"est": (x, x), "cost": None, "it": k + 1}

        def finish(self, x, cur_prior, cur_params, iterations):
            """The threshold BART applies after the loop."""
            k = max(iterations - 1, 0)
            return cur_prior.prox(
                x, cur_params.get("g_param"), gamma=_scale(cur_params, k) * _tau(cur_params, k)
            )

    return ISTIteration


def _fista() -> type:
    _, _, OptimIterator = _classes()

    class FISTAIteration(OptimIterator):
        r"""BART's fast iterative soft thresholding, one step of it.

        ``italgos.c``'s ``fista``: the same as :class:`ISTIteration` with
        Nesterov's ravine step between the threshold and the residual.  The
        momentum is BART's ``fista_formula``,

        ``t <- (p + sqrt(q + r t^2)) / 2``

        with ``(p, q, r) = (1, 1, 4)`` unless ``--fista_pqr`` says otherwise,
        and the ravine itself carries the previous thresholded iterate, which
        is what ``est``'s second half holds here.
        """

        def __init__(self, **kwargs):
            kwargs.setdefault("has_cost", False)
            super().__init__(**kwargs)

        def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *a, **kw):
            x, previous = X["est"]
            k = X.get("it", 0)
            t = X.get("t", 1.0)
            tau = _tau(cur_params, k)
            p, q, r = cur_params.get("pqr", (1.0, 1.0, 4.0))

            z = cur_prior.prox(
                x,
                cur_params.get("g_param"),
                gamma=_scale(cur_params, k) * tau * cur_params.get("alpha", 1.0),
            )

            # `ravine`: swap the two, then two axpys.  Written as the axpys
            # rather than as the combination they add up to, because
            # `x + c * x` and `(1 + c) * x` are not the same float32 number
            # and the iteration is held against BART's to the bit.
            told, t = t, _formula(p, q, r, t)
            before, after = _ravine(told, t)
            x = previous
            x = x + before * x
            x = x + after * z

            x = x - tau * cur_data_fidelity.grad(x, y, physics)

            return {"est": (x, z), "cost": None, "it": k + 1, "t": t}

        def finish(self, x, cur_prior, cur_params, iterations):
            k = max(iterations - 1, 0)
            return cur_prior.prox(
                x,
                cur_params.get("g_param"),
                gamma=_scale(cur_params, k) * _tau(cur_params, k) * cur_params.get("alpha", 1.0),
            )

    return FISTAIteration


def _single(value: float) -> float:
    """``value`` as the single-precision number BART would have held.

    Every scalar in these iterations is a C ``float`` unless BART declares it
    a ``double`` -- which it does for the residual accumulators in ``admm``
    and nowhere else.  Python has only doubles, so each scalar is rounded
    where the library would have rounded it; a scalar worked out in a double
    and rounded once at the end is a different number, which is what made the
    ravine's coefficients diverge at the thirteenth iteration.
    """
    return float(np.float32(value))


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """``vecops.c``'s ``dot``: the products in single precision, summed in a
    double, over the real and imaginary parts as one long vector."""
    x = torch.view_as_real(a) if a.is_complex() else a
    z = torch.view_as_real(b) if b.is_complex() else b
    return float((x * z).double().sum())


def _norm(x: torch.Tensor) -> float:
    """``vecops.c``'s ``norm``, which is not a single-precision norm.

    BART squares the components in single precision and then sums and roots
    them in a double, and hands back the double.  Where the result is put in a
    ``float`` the difference is rounded away; where it is squared again --
    which is what the primal-dual step adaptation does to it -- it is not, and
    it reaches the iterate.
    """
    parts = torch.view_as_real(x) if x.is_complex() else x
    return float(torch.sqrt((parts * parts).double().sum()))


def _ravine(told: float, t: float) -> tuple[float, float]:
    """The two coefficients ``ravine`` makes, each operation in single precision.

    ``(1.f - tfo) / ft - 1.f`` is three single-precision operations in C, and
    working the same expression out in a double and rounding once at the end
    is not the same number.  It takes thirteen iterations for the difference
    to reach the answer, which is the sort of thing that only a comparison
    against the library finds.
    """
    one = np.float32(1.0)
    t32, told32 = np.float32(t), np.float32(told)
    return float((one - told32) / t32 - one), float((told32 - one) / t32 + one)


def _formula(p: float, q: float, r: float, t: float) -> float:
    """``fista_formula``: ``(p + sqrtf(q + r t^2)) / 2``, in single precision.

    Whether ``q + r * t * t`` is one rounding or two is the compiler's choice,
    not the source's: clang contracts it into a fused multiply-add where the
    hardware has one, which arm64 does and the x86-64 baseline does not, and a
    fused multiply-add does not round the product.  Written out here as the
    two roundings, which is what BART computes on x86-64.

    It makes no difference to ``pics``.  With BART's own ``r = 4`` the two
    forms agree at every step of the recurrence -- a test measures that -- so
    the default is the same bits on either.  ``--fista_pqr`` with some other
    ``r`` is where they part, and there a solve on arm64 can differ from this
    one in the last places.
    """
    t32 = np.float32(t)
    inner = np.float32(np.float32(q) + np.float32(r) * t32 * t32)
    return float((np.float32(p) + np.sqrt(inner)) / np.float32(2.0))


def _tau(params: dict, k: int) -> float:
    """BART's step at iteration ``k``: the step over the largest eigenvalue,
    halved by ``hogwild`` after 10 steps, then 20, then 40."""
    tau = _single(params["stepsize"])
    if not params.get("hogwild", False):
        return tau

    seen, period = 0, 10
    for _ in range(k + 1):
        seen += 1
        if seen == period:
            seen, period, tau = 0, period * 2, _single(tau / 2)
    return tau


def _scale(params: dict, k: int) -> float:
    """BART's continuation: the threshold decays geometrically to
    ``continuation`` over the run, and stays put at the default of one."""
    c = params.get("continuation", 1.0)
    if 1.0 == c:
        return 1.0
    a = np.log(np.float32(c)) / np.float32(params["maxiter"])
    return float(np.exp(a * np.float32(k)))


def _admm() -> type:
    _, _, OptimIterator = _classes()

    class ADMMIteration(OptimIterator):
        r"""BART's alternating direction method of multipliers, one step of it.

        ``admm.c``'s ``admm``, which solves

        ``min_x 0.5 ||A x - y||^2 + sum_j f_j(G_j x - b_j)``

        for arbitrary convex ``f_j``.  Each step solves for ``x`` by conjugate
        gradients on ``A^H A + rho sum_j G_j^H G_j``, warm-started from the
        iterate before it, then updates each term's split variable and dual::

            rhs = A^H y + rho sum_j G_j^H (z_j - u_j + b_j)
            x   = cg(rhs, from x)
            w_j = alpha G_j x + (1 - alpha) (z_j + b_j) + u_j - b_j
            z_j = prox_j(w_j, lambda / rho)
            u_j = w_j - z_j

        with ``alpha = 1.6``, BART's over-relaxation, and Boyd's primal and
        dual residuals deciding when to stop and -- with ``dynamic_rho`` --
        how ``rho`` moves.

        The inner solve is BART's own conjugate gradients over an operator
        whose normal is the one above, so the encoding is asked for *its*
        normal: a Toeplitz encoding stays one inside every step.

        Parameters
        ----------
        terms : sequence of Regularizer
            The ``f_j``, each with the transform and bias it carries.
        image_shape : tuple of int
            What the terms are configured for.
        biases : sequence of tensor, optional
            The ``b_j``, each of its term's transformed shape.

        Notes
        -----
        ``maxiter`` is a budget on applications of the normal operator rather
        than a count of outer steps, which is BART's rule and a surprising
        one: ``admm`` breaks when ``nr_invokes > maxiter``, and ``nr_invokes``
        counts conjugate-gradient iterations across the whole run.  Thirty
        with ten inner iterations is about five outer steps, not thirty.
        """

        def __init__(self, terms, image_shape, biases=None, **kwargs):
            kwargs.setdefault("has_cost", False)
            super().__init__(**kwargs)
            self.terms = list(terms)
            self.image_shape = tuple(image_shape)
            self.biases = list(biases) if biases is not None else [None] * len(self.terms)
            if len(self.biases) != len(self.terms):
                raise ValueError("one bias per term, or none at all")
            self._invokes = 0

        # --- the pieces of a term ---------------------------------------------

        def _forward(self, term, x):
            return term.apply_transform(x, self.image_shape)

        def _adjoint(self, term, v):
            return term.apply_transform(v, self.image_shape, mode="adjoint")

        def _normal(self, term, x):
            return term.apply_transform(x, self.image_shape, mode="normal")

        def _shape(self, term):
            return term.prox_shape(self.image_shape)

        # --- the x update ------------------------------------------------------

        def _solve_x(self, x, rhs, rho, physics, params, first=False):
            """Conjugate gradients on ``A^H A + rho sum_j G_j^H G_j``, from ``x``.

            BART's ``cg_xupdate``: the same operator, the same warm start, and
            the same stopping rule -- ``cg_eps`` times the norm of the right
            hand side.  It counts the applications as it goes, because that
            is the budget ``maxiter`` actually is.
            """
            from bartorch.linop import Callback, Identity
            from bartorch.linop.base import _WithNormal
            from bartorch.optim.linear import CG

            if 0.0 == float(torch.linalg.vector_norm(rhs)):
                return x

            op = getattr(physics, "op", None)

            cclambda = params.get("cclambda", 0.0)

            def apply(v):
                # `admm_normaleq`'s order, which is not the obvious one: the
                # terms are summed first, scaled by `rho` as they go, and the
                # encoding's normal -- with the quadratic weight folded into
                # it, as `normaleq_l2_apply` does -- is added last.  With one
                # term the two orders agree bit for bit; with two they do not.
                out = None
                for term in self.terms:
                    contribution = rho * self._normal(term, v)
                    out = contribution if out is None else out + contribution
                normal = (
                    _batched(op.normal, v, op.ishape)
                    if op is not None
                    else physics.A_adjoint(physics.A(v))
                )
                if cclambda:
                    normal = normal + cclambda * v
                return normal if out is None else out + normal

            shape = tuple(x.shape)
            normal = Callback(shape, shape, apply, apply, apply)
            budget = params.get("cg_maxiter", 10)
            if first and params.get("cg_maxiter_first") is not None:
                # riesling's `iters0`: more inner iterations on the first outer
                # step, where there is no warm start to build on, and fewer
                # after.  BART has one budget for every step; this is the one
                # thing that implementation has that BART's does not.
                budget = params["cg_maxiter_first"]
            solver = CG(maxiter=budget, tol=params.get("cg_eps", 1e-3))

            steps: list[int] = []
            out = solver(rhs, _WithNormal(Identity(shape), normal), x0=x, steps=steps)

            # `nr_invokes` is the conjugate-gradient iterations, which the
            # library now reports rather than leaving to be inferred.
            self._invokes += steps[0]
            return out

        # --- one step ----------------------------------------------------------

        def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *a, **kw):
            x = X["est"][0]
            # `float rho, tau` in `admm.c`; the accumulators below are its
            # `double n1, n2, n3, r_scaling, s_scaling`, kept as such.
            rho = _single(X.get("rho", cur_params.get("rho", 0.5)))
            tau = _single(X.get("tau", cur_params.get("tau", 2.0)))
            alpha = cur_params.get("alpha", 1.6)
            lam = cur_params.get("lambda", 1.0)
            fast = cur_params.get("fast", False)

            z = X.get("z")
            u = X.get("u")
            if z is None:
                z = [
                    torch.zeros(self._shape(t), dtype=x.dtype, device=x.device) for t in self.terms
                ]
                u = [torch.zeros_like(zj) for zj in z]

            adjoint = cur_data_fidelity._adjoint_data(y, physics)

            rhs = torch.zeros_like(x)
            for j, term in enumerate(self.terms):
                r = z[j] - u[j]
                if self.biases[j] is not None:
                    r = r + self.biases[j]
                rhs = rhs + self._adjoint(term, r)
            rhs = rho * rhs + adjoint

            x = self._solve_x(x, rhs, rho, physics, cur_params, first=0 == X.get("it", 0))

            n1 = n2 = r_sq = 0.0
            s = torch.zeros_like(x)
            gh_usum = torch.zeros_like(x)

            for j, term in enumerate(self.terms):
                bias = self.biases[j]
                gx = self._forward(term, x)
                z_old = z[j]

                if not fast:
                    residual = gx
                    n1 += _norm(residual) ** 2
                    gx = alpha * gx + (1.0 - alpha) * z[j]
                    if bias is not None:
                        gx = gx + (1.0 - alpha) * bias

                w = gx + u[j]
                if bias is not None:
                    w = w - bias

                z[j] = term.prox(w, lam / rho, image_shape=self.image_shape) if rho else w
                u[j] = w - z[j]

                if not fast:
                    r = residual - z[j]
                    if bias is not None:
                        r = r - bias
                    # `float r_norm` against `double n1, n2`: the primal
                    # residual is accumulated in single precision and the
                    # scalings in double, which is what `admm.c` declares.
                    r_sq = _single(r_sq + _norm(r) ** 2)
                    s = s + self._adjoint(term, z[j] - z_old)
                    gh_usum = gh_usum + self._adjoint(term, u[j])
                    n2 += _norm(z[j]) ** 2

            done = False
            if not fast:
                r_norm = _single(math.sqrt(r_sq))
                s_norm = _single(rho * _norm(s))
                n3 = sum(_norm(b) ** 2 for b in self.biases if b is not None)
                r_scaling = math.sqrt(max(n1, n2, n3))
                s_scaling = rho * _norm(gh_usum)

                # BART counts real numbers, which is twice the complex ones.
                m = 2 * sum(math.prod(self._shape(t)) for t in self.terms)
                n = 2 * math.prod(tuple(x.shape))
                # Zero by default, which is what `italgo_config` sets them to:
                # `iter_admm_defaults` carries 1e-4 and 1e-3, and `pics`
                # overrides both, so the residual test never fires and the
                # budget is the only thing that stops the iteration.
                abstol = cur_params.get("abstol", 0.0)
                reltol = cur_params.get("reltol", 0.0)
                eps_pri = _single(abstol * math.sqrt(m) + reltol * r_scaling)
                eps_dual = _single(abstol * math.sqrt(n) + reltol * s_scaling)

                done = self._spent(cur_params, X) or (r_norm < eps_pri and s_norm < eps_dual)
                rho, tau = self._adapt(
                    cur_params, rho, tau, r_norm, s_norm, r_scaling, s_scaling, u
                )
            else:
                done = self._spent(cur_params, X)
                rho, tau = self._adapt(cur_params, rho, tau, 0.0, 0.0, 1.0, 1.0, u)

            return {
                "est": (x, x),
                "cost": None,
                "z": z,
                "u": u,
                "rho": rho,
                "tau": tau,
                "done": done,
                "it": X.get("it", 0) + 1,
            }

        def _spent(self, params, X) -> bool:
            """Whether BART would stop here, budget-wise.

            Two limits, not one, and missing the second is what made this hard
            to read off the source: `admm`'s own loop runs at most `maxiter`
            times, *and* it breaks when `nr_invokes > maxiter`, where
            `nr_invokes` is the conjugate-gradient iterations across the whole
            run.  Whichever comes first.

            On a well-conditioned problem the inner solve takes one iteration
            a step, so the loop bound is what stops it and `maxiter` does look
            like a count of steps.  On a hard one the iterations pile up and
            the budget stops it long before.
            """
            maxiter = params["maxiter"]
            return (X.get("it", 0) + 1 >= maxiter) or (self._invokes > maxiter)

        def _adapt(self, params, rho, tau, r_norm, s_norm, r_scaling, s_scaling, u):
            """BART's ``tau`` and ``rho`` moves, and hogwild's doubling."""
            sc = 1.0

            if params.get("dynamic_tau", False):
                tau_max = params.get("tau_max", 20.0)
                # `sqrt(r_norm / s_norm)` with both at zero -- which is what
                # `fast` leaves them at -- is a NaN, and every comparison
                # below is then false, so `tau` goes to its ceiling.  BART
                # does not guard it either.
                # `sqrt(r_norm / s_norm)` over two floats: the division is a
                # single-precision one, and only the root is taken in double.
                t = math.sqrt(_single(r_norm / s_norm)) if s_norm else float("nan")
                if tau_max > t >= 1.0:
                    tau = _single(t)
                elif 1.0 > t > 1.0 / tau_max:
                    tau = _single(1.0 / t)
                else:
                    tau = _single(tau_max)

            if params.get("dynamic_rho", False):
                r, s = r_norm, s_norm
                if params.get("relative_norm", False):
                    r, s = r / r_scaling, s / s_scaling
                mu = params.get("mu", 3.0)
                if r > mu * s:
                    sc = tau
                elif s > mu * r:
                    sc = _single(1.0 / tau)

            if params.get("hogwild", False):
                self._hw_k = getattr(self, "_hw_k", 0) + 1
                self._hw_K = getattr(self, "_hw_K", 1)
                if self._hw_k == self._hw_K:
                    self._hw_k, self._hw_K, sc = 0, self._hw_K * 2, 2.0

            if 1.0 != sc:
                rho = _single(rho * sc)
                # `smul(z_dims[j], 1. / sc, u[j], u[j])`: the reciprocal once,
                # as a float, and the vector scaled by it.
                back = _single(1.0 / sc)
                for j in range(len(u)):
                    u[j] = back * u[j]

            return rho, tau

        def restart(self):
            """Forget the budget already spent, for a fresh run."""
            self._invokes = 0
            self._hw_k, self._hw_K = 0, 1

    return ADMMIteration


def _pridu() -> type:
    _, _, OptimIterator = _classes()

    class PRIDUIteration(OptimIterator):
        r"""BART's primal-dual iteration, one step of it.

        ``italgos.c``'s ``chambolle_pock``, which ``pics --pridu`` runs.  The
        data term is carried as its own dual variable rather than
        differentiated: ``A^H u`` is updated through the resolvent

        ``A^H u <- (sigma A^H A x_avg + A^H u - sigma A^H y) / (1 + sigma)``

        and each regularization term gets a dual of its own, updated through
        its proximal operator's conjugate.  The primal step is a descent on
        the duals followed by ``prox2``, and ``x_avg`` extrapolates.

        ``pics`` takes ``sigma = sqrt(step) * ratio`` and
        ``tau = sqrt(step) / ratio`` with ``theta = 1``, and ``hogwild`` there
        is a decay of 0.95 a step rather than a halving.

        The first term stands apart, as it does in ``iter2_chambolle_pock``:
        a term whose transform is the identity becomes the primal ``prox2``
        and the rest become duals.  Without such a term ``prox2`` is the
        identity, which is what ``prox_zero_create`` is.

        Notes
        -----
        This is the one iteration here whose answer depends on how BART was
        compiled.  ``vecops.c`` has a single kernel behind ``axpy``, ``xpay``
        and ``axpbz``, ``dst[i] = a1 * src1[i] + a2 * src2[i]``, and clang
        folds the first product into the add where the hardware has a fused
        multiply-add -- arm64 does, the x86-64 baseline does not.  A fused
        multiply-add does not round the product, and torch cannot fuse across
        two kernels.

        Every other iteration escapes it because its updates are ``axpy``,
        whose ``a1`` is one: folding an exact product in changes nothing.  The
        data term's resolvent here is an ``xpay`` and an ``axpbz`` with two
        real coefficients, so on a platform that folds them this iteration is
        within a few times 1e-7 of the library rather than the same bits.
        """

        def __init__(self, terms, image_shape, primal=None, **kwargs):
            kwargs.setdefault("has_cost", False)
            super().__init__(**kwargs)
            self.terms = list(terms)
            self.primal = primal
            self.image_shape = tuple(image_shape)

        def _prox2(self, x, gamma):
            if self.primal is None:
                return x
            return self.primal.prox(x, gamma, image_shape=self.image_shape)

        def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *a, **kw):
            x = X["est"][0]
            avg = X.get("avg", x)
            duals = X.get("duals")
            adjoint_dual = X.get("adjoint_dual", torch.zeros_like(x))
            k = X.get("it", 0)

            # `float sigma, tau` in `chambolle_pock`, and so are the residuals.
            sigma = _single(X.get("sigma", cur_params["sigma"]))
            tau = _single(X.get("tau", cur_params["tau"]))
            theta = cur_params.get("theta", 1.0)
            alpha = cur_params.get("alpha", 1.0)
            decay = cur_params.get("decay", 1.0)
            # `float lambda = (float)pow(decay, i)`, from a float32 `decay`:
            # the power is taken in double and rounded once at the end.
            lam = 1.0 if 1.0 == decay else float(np.float32(float(np.float32(decay)) ** k))

            if duals is None:
                duals = [
                    torch.zeros(t.prox_shape(self.image_shape), dtype=x.dtype, device=x.device)
                    for t in self.terms
                ]

            # The data term's dual, through its resolvent.
            previous = adjoint_dual
            step = sigma * cur_data_fidelity.normal(avg, physics) + adjoint_dual
            # `axpbz(Ahu_new, 1. / (1. + sigma), Ahu_old, -1. * sigma / (1. + sigma), xadj)`:
            # both coefficients are worked out in a double and rounded to the
            # float each vector is scaled by.
            keep = _single(1.0 / (1.0 + sigma))
            pull = _single(-1.0 * sigma / (1.0 + sigma))
            fresh = keep * step + pull * cur_data_fidelity._adjoint_data(y, physics)
            adjoint_dual = lam * fresh + (1.0 - lam) * previous
            change = adjoint_dual - previous
            moved = float(torch.real((change.conj() * change).sum()))

            # Each regularization term's, through the conjugate of its prox.
            for j, term in enumerate(self.terms):
                # `axpy(u_old, 1. / sigma, u[j])`: the reciprocal is worked
                # out once, in a double, and rounded to the float the vector
                # is scaled by -- not a division of the vector.
                over = term.apply_transform(avg, self.image_shape) + _single(1.0 / sigma) * duals[j]
                thresholded = term.prox(over, alpha / sigma, image_shape=self.image_shape)
                fresh_j = sigma * over - sigma * thresholded
                was = duals[j]
                duals[j] = lam * fresh_j + (1.0 - lam) * was
                moved += float(torch.real(((duals[j] - was).conj() * (duals[j] - was)).sum()))

            # The primal step.
            previous_x = x
            x = x - tau * adjoint_dual
            for j, term in enumerate(self.terms):
                x = x - tau * term.apply_transform(duals[j], self.image_shape, mode="adjoint")
            x = lam * self._prox2(x, tau * alpha) + (1.0 - lam) * previous_x

            # `res2` is measured against the step `tau` had before the
            # adaptation, as it is in `chambolle_pock`.
            res2 = _single(_single(math.sqrt(max(moved, 0.0))) / tau)
            res1 = _single(_single(_norm(x - previous_x)) / sigma)

            if cur_params.get("adaptive_step", False):
                sigma, tau = self._adapt(
                    x - previous_x, sigma, tau, cur_params, cur_data_fidelity, physics
                )

            avg = (1.0 + theta) * x - theta * previous_x

            # `iter2_chambolle_pock` leaves `eps` at one, so the tolerance is
            # absolute rather than relative to the data -- unlike every other
            # iteration here, where it is scaled by the norm of `A^H y`.
            epsilon = cur_params.get("tol", 1e-4)

            return {
                "est": (x, x),
                "cost": None,
                "avg": avg,
                "duals": duals,
                "adjoint_dual": adjoint_dual,
                "sigma": sigma,
                "tau": tau,
                "done": epsilon > (res1 + res2),
                "it": k + 1,
            }

        def _adapt(self, delta, sigma, tau, params, cur_data_fidelity, physics):
            """BART's step adaptation: the ratio of the move to what the
            operator makes of it, clipped just under ``sqrt(sigma tau)``."""
            # `float norm_Kx`, and each `+=` rounds back to a float: the
            # terms are summed one at a time, not worked out together and
            # rounded once.  With a single dual term the two are the same
            # number; with a dual and the data term they are not.
            squared = 0.0
            for term in self.terms:
                moved = term.apply_transform(delta, self.image_shape)
                squared = _single(squared + _norm(moved) ** 2)
            normal = cur_data_fidelity.normal(delta, physics)
            squared = _single(squared + _dot(normal, delta))

            norm_kx = _single(math.sqrt(max(squared, 0.0)))
            if 0.0 == norm_kx:
                return sigma, tau

            # Every one of these is a single-precision operation in C, down to
            # the literal: `0.95f` is not 0.95, and `sqrtf(sigma * tau)` roots
            # a product that has already been rounded.
            ratio = _single(_single(_norm(delta)) / norm_kx)
            root = _single(math.sqrt(_single(sigma * tau)))
            threshold = _single(_single(0.95) * root)
            if 0.0 != ratio < root:
                chosen = min(threshold, ratio)
            else:
                chosen = root

            r = _single(params.get("sigma_tau_ratio", 1.0))
            return _single(chosen * r), _single(chosen / r)

    return PRIDUIteration


_BUILDERS = {
    "ADMMIteration": _admm,
    "PRIDUIteration": _pridu,
    "NormalEquations": _normal_equations,
    "TermPrior": _term_prior,
    "ISTIteration": _ist,
    "FISTAIteration": _fista,
    "AsTerm": _as_term,
}


def __getattr__(name: str):
    if name in _BUILDERS:
        return _built(name)
    raise AttributeError(name)
