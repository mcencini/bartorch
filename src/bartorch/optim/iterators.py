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

``deepinv`` is imported on first use, so it stays an optional dependency.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Built on first use, so that importing this module does not import deepinv;
# `__getattr__` below is what resolves them.
__all__ = [  # noqa: F822
    "FISTAIteration",
    "ISTIteration",
    "NormalEquations",
    "TermPrior",
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
            "deepinv is required for bartorch.optim's iterations; "
            "install it with `pip install 'bartorch[deepinv]'`"
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

        def __init__(self):
            super().__init__()
            self._adjoint: tuple | None = None

        def _adjoint_data(self, y: torch.Tensor, physics) -> torch.Tensor:
            key = (id(y), y._version, y.shape)
            if self._adjoint is None or self._adjoint[0] != key:
                op = physics.op
                self._adjoint = (key, _batched(op.adjoint, y, op.oshape))
            return self._adjoint[1]

        def grad(self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs):
            op = getattr(physics, "op", None)
            if op is None:
                return physics.A_adjoint(physics.A(x) - y)
            return _batched(op.normal, x, op.ishape) - self._adjoint_data(y, physics)

        def fn(self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs):
            residual = physics.A(x) - y
            return 0.5 * residual.flatten(1).abs().pow(2).sum(-1)

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
    """``fista_formula``: ``(p + sqrtf(q + r t^2)) / 2``, in single precision."""
    t32 = np.float32(t)
    inner = np.float32(np.float32(q) + np.float32(r) * t32 * t32)
    return float((np.float32(p) + np.sqrt(inner)) / np.float32(2.0))


def _tau(params: dict, k: int) -> float:
    """BART's step at iteration ``k``: the step over the largest eigenvalue,
    halved by ``hogwild`` after 10 steps, then 20, then 40."""
    tau = params["stepsize"]
    if not params.get("hogwild", False):
        return tau

    seen, period = 0, 10
    for _ in range(k + 1):
        seen += 1
        if seen == period:
            seen, period, tau = 0, period * 2, tau / 2
    return tau


def _scale(params: dict, k: int) -> float:
    """BART's continuation: the threshold decays geometrically to
    ``continuation`` over the run, and stays put at the default of one."""
    c = params.get("continuation", 1.0)
    if 1.0 == c:
        return 1.0
    a = np.float32(math.log(c)) / np.float32(params["maxiter"])
    return float(np.exp(a * np.float32(k)))


_BUILDERS = {
    "NormalEquations": _normal_equations,
    "TermPrior": _term_prior,
    "ISTIteration": _ist,
    "FISTAIteration": _fista,
}


def __getattr__(name: str):
    if name in _BUILDERS:
        return _built(name)
    raise AttributeError(name)
