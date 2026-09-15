"""Regularized least squares by the iterations ``pics`` runs.

Each solver hands BART's ``lsqr2`` the encoding, the operators BART built for
the :mod:`bartorch.priors` terms, and its settings, through the same
``italgo_config`` call ``pics`` makes.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand
from bartorch.priors.base import Regularizer, _as_terms
from bartorch.priors.terms import L2

__all__ = ["ADMM", "CG", "PRIDU", "FISTA", "IST", "NIHT", "Tikhonov"]


@dataclasses.dataclass(frozen=True)
class Tikhonov:
    """A quadratic penalty ``weight * ||operator x - bias||^2``.

    What generalized Tikhonov regularization is, and what
    :class:`CG` minimizes alongside the data term.  Without an operator the
    penalty is on the image itself, and without a bias it is on its size
    rather than its distance from something.

    Every combination is still a least-squares problem, so it is still
    conjugate gradients that solves it: the terms are stacked under the
    encoding and the normal operator is the sum of the terms' own, which is
    what keeps a Toeplitz encoding's normal the convolution it was.

    Parameters
    ----------
    weight : float
        The weight, not its square root.  Must not be negative.
    operator : LinearOperator, optional
        What the penalty is on, mapping the image somewhere.  By default the
        image itself.
    bias : tensor, optional
        What the penalty pulls towards, of the operator's codomain shape.  By
        default zero, which is the ordinary penalty on size.

    Examples
    --------
    Pull towards a prior image rather than towards zero:

    >>> CG(terms=Tikhonov(0.1, bias=prior))(y, A)

    Penalize the first differences, which is quadratic total variation:

    >>> CG(terms=Tikhonov(0.1, operator=linop.Gradient(A.ishape, (-1, -2))))(y, A)
    """

    weight: float
    operator: object | None = None
    bias: torch.Tensor | None = None

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(
                f"a quadratic penalty has a weight of at least zero, not {self.weight}"
            )

    def _operator(self, ishape: tuple[int, ...]):
        """This term's operator, or the identity on the image."""
        if self.operator is not None:
            return self.operator

        from bartorch.linop import Identity

        return Identity(ishape)


Terms = Tikhonov | Iterable["Tikhonov"] | None


def _as_quadratics(terms: Terms) -> list[Tikhonov]:
    """``terms`` -- None, one term or an iterable of them -- as a list of terms."""
    if terms is None:
        return []
    if isinstance(terms, Tikhonov):
        return [terms]

    def refuse(what):
        return TypeError(
            f"CG takes Tikhonov terms, not {what!r}; a term with a proximal operator "
            "rather than a quadratic one goes to a solver that has one, such as FISTA"
        )

    try:
        out = list(terms)
    except TypeError:
        raise refuse(terms) from None
    for term in out:
        if not isinstance(term, Tikhonov):
            raise refuse(term)
    return out


def _stacked(A, y: torch.Tensor, terms: Sequence[Tikhonov]):
    """``(A~, y~)`` for ``min ||A x - y||^2 + sum_i w_i ||G_i x - b_i||^2``.

    Written as one least-squares problem, which is what conjugate gradients
    solves::

        A~ = [A; sqrt(w_1) G_1; ...]        y~ = [y; sqrt(w_1) b_1; ...]

    The codomains have nothing in common -- samples against images against
    differences -- so each is read as the line of numbers it is and they are
    laid end to end, which is what ``linop_stack_cod`` does and what makes the
    normal of the whole the sum of the parts' own normals.

    That last part matters.  A^H A for a Toeplitz encoding is a
    convolution rather than two transforms, and it stays one here: the normal
    is built as ``A.gram() + sum_i w_i G_i.gram()`` and attached with
    ``linop_from_ops``, so the encoding is asked for its normal rather than
    for its forward and its adjoint.
    """
    from bartorch.linop import Reshape, concatenate
    from bartorch.linop.base import _WithNormal

    op = A._bart()
    y = as_operand(y, op.oshape, "y")

    def flat(operator):
        size = math.prod(operator.oshape)
        return Reshape((size,), operator.oshape) @ operator

    pieces = [flat(A)]
    data = [y.reshape(-1)]
    normal = A.gram()

    for term in terms:
        G = term._operator(op.ishape)
        if G.ishape != op.ishape:
            raise ValueError(f"a term over {G.ishape} does not fit an encoding over {op.ishape}")
        # The weight meets a complex64 tensor, so its square root is taken in
        # the precision that tensor is held at.
        root = float(np.float32(math.sqrt(term.weight)))
        pieces.append(flat(root * G) if 1.0 != root else flat(G))

        if term.bias is None:
            data.append(torch.zeros(math.prod(G.oshape), dtype=y.dtype, device=y.device))
        else:
            data.append(root * as_operand(term.bias, G.oshape, "bias").reshape(-1))

        normal = normal + (term.weight * G.gram() if 1.0 != term.weight else G.gram())

    return _WithNormal(concatenate(pieces), normal), torch.cat(data)


Regularizers = Regularizer | Iterable[Regularizer] | None


def maxeigen(A, terms: Terms = None, *, cclambda: float = 0.0, iterations: int = 30) -> float:
    """BART's estimate of the largest eigenvalue of the operator a step divides by.

    What ``pics -e`` asks for.  It is a power iteration from a random start,
    so it draws on BART's own generator: a loop written outside the library
    has to ask for it here, at the point in the sequence the library would
    have asked, or the draws that follow -- a wavelet term's cycle spinning,
    say -- are different ones.

    Parameters
    ----------
    A : LinearOperator
        The encoding.  Its normal is the operator, with ``cclambda`` on the
        diagonal, as ``lsqr`` builds it.
    terms : Regularizer or iterable of Regularizer, optional
        Terms whose transforms are added to it.  That is what the primal-dual
        iteration estimates over; the proximal ones take the encoding alone.
    cclambda : float
        The quadratic weight (``pics -q``).
    iterations : int
        Power iterations; BART takes thirty.

    Returns
    -------
    float
    """
    op = A._bart()
    _ensure_ready()
    lib = library()

    handles = [term.build(op.ishape) for term in _as_terms(terms)]
    value = _marshal.double_out()
    with _lock, _on_device(op.device or torch.device("cpu")):
        code = lib.bartorch_maxeigen(
            op._h.ptr,
            float(cclambda),
            len(handles),
            _marshal.pointers(handles) if handles else None,
            int(iterations),
            _marshal.by_reference(value),
        )
    if code != 0:
        raise BartError("the largest eigenvalue could not be estimated")
    return float(value.value)


def _solve(
    A,
    y: torch.Tensor,
    x0: torch.Tensor | None,
    algorithm: str,
    terms: list[Regularizer],
    *,
    maxiter: int,
    cclambda: float,
    step: float = -1.0,
    eigen: bool = False,
    hogwild: bool = False,
    rho: float = -1.0,
    cg_maxiter: int = 0,
    cg_tol: float = 0.0,
    dynamic_rho: bool = False,
    dynamic_tau: bool = False,
    relative_norm: bool = False,
    fast: bool = False,
    pqr: tuple[float, float, float] | None = None,
    sigma_tau_ratio: float = 1.0,
    adaptive_step: bool = False,
    precond=None,
    steps: list | None = None,
) -> torch.Tensor:
    """Run ``bartorch_solve``.  A negative ``step`` or ``rho``, a zero
    ``cg_maxiter`` and ``pqr=None`` keep BART's defaults.

    ``steps``, when given, has the number of iterations the algorithm took
    appended to it."""
    op = A._bart()
    y = as_operand(y, op.oshape, "y")
    if x0 is None:
        x = torch.zeros(op.ishape, dtype=torch.complex64, device=y.device)
    else:
        x = as_operand(x0, op.ishape, "x0").clone()

    _ensure_ready()
    lib = library()
    ndim = len(op.ishape)
    flags = [term._flags(ndim) for term in terms]
    # A term that adds unknowns cannot be built on its own -- the offsets its
    # transforms sit at are worked out across the whole set -- so on that path
    # the solve configures the set itself and there is nothing to hand over.
    extends = _extending(terms)
    for term in terms:
        term._check(ndim)
    handles = [] if extends else [term.build(op.ishape) for term in terms]
    p, q, r = pqr if pqr is not None else (-1.0, -1.0, -1.0)
    # The preconditioner is one more BART operator, and it has to outlive the
    # call; the wrapper a Python-defined one produces is kept here for that.
    conditioner = None if precond is None else precond._bart()
    if conditioner is not None and (
        conditioner.ishape != op.ishape or conditioner.oshape != op.ishape
    ):
        raise ValueError(
            f"a preconditioner maps the image to itself, so it is {op.ishape} to "
            f"{op.ishape}, not {conditioner.ishape} to {conditioner.oshape}"
        )

    # `opt_reg_configure` takes one block size, one wavelet family and one
    # shift mode for the whole set, and reaches for them only on the path that
    # configures the set -- which is the path an extending term forces.
    block, family, shift_mode = _shared_options(terms)
    alpha, gamma = _shared_pairs(terms)

    counter = _marshal.long_out()
    with _lock, _on_device(op.device or y.device):
        code = lib.bartorch_solve(
            op._h.ptr,
            algorithm.encode(),
            _marshal.argv([term.kind for term in terms]) if terms else None,
            _marshal.longs([f for f, _ in flags]) if terms else None,
            _marshal.longs([j for _, j in flags]) if terms else None,
            _marshal.floats([term.weight for term in terms]) if terms else None,
            _marshal.ints([term.count for term in terms]) if terms else None,
            _marshal.pointers(handles) if handles else None,
            len(terms),
            float(cclambda),
            int(maxiter),
            float(step),
            int(eigen),
            int(hogwild),
            float(rho),
            int(cg_maxiter),
            float(cg_tol),
            int(dynamic_rho),
            int(dynamic_tau),
            int(relative_norm),
            int(fast),
            float(p),
            float(q),
            float(r),
            float(sigma_tau_ratio),
            int(adaptive_step),
            int(x0 is not None),
            None if conditioner is None else conditioner._h.ptr,
            block,
            family.encode(),
            shift_mode,
            _marshal.floats(alpha),
            _marshal.floats(gamma),
            x.data_ptr(),
            y.data_ptr(),
            _marshal.by_reference(counter) if steps is not None else None,
        )
    if code != 0:
        said = lib.bartorch_solve_error(code).decode(errors="replace")
        raise BartError(f"the solve failed: {said}")
    if steps is not None:
        steps.append(int(counter.value))
    return x


def _start(A, y: torch.Tensor, x0: torch.Tensor | None, terms: Sequence[Regularizer] = ()):
    """The operands a loop written here starts from: the BART operator, the
    data, and the image the iteration walks.

    The terms are rewound, which is what ``bartorch_solve`` does before it
    hands them over: a wavelet threshold's cycle spinning comes from a
    generator of its own, and a term kept across solves would otherwise carry
    the last solve's draws into the next.
    """
    op = A._bart()
    for term in terms:
        term.rewind(op.ishape)
    y = as_operand(y, op.oshape, "y")
    x = (
        torch.zeros(op.ishape, dtype=torch.complex64, device=y.device)
        if x0 is None
        else as_operand(x0, op.ishape, "x0").clone()
    )
    return op, y, x


def _zeros(y: torch.Tensor, image_shape: tuple[int, ...]) -> dict:
    """Where a network starts: the image BART starts at, which is zero.

    ``deepinv`` starts an optimizer at ``A^H y`` instead.  Starting where the
    solver starts is what makes a network with nothing trainable in it answer
    with the solver's numbers; ``custom_init=None`` restores ``deepinv``'s
    start, which a network about to be trained may prefer::

        solver.unrolled(shape, custom_init=None)
    """
    batch = y.shape[0] if y.ndim == len(image_shape) + 1 else 1
    zeros = torch.zeros((batch, *image_shape), dtype=torch.complex64, device=y.device)
    return {"est": (zeros, zeros)}


def _empty(adjoint: torch.Tensor) -> bool:
    """``checkeps``: BART declines to iterate on data whose adjoint has no
    norm, or whose norm is not a normal number, and leaves the image be."""
    eps = float(np.float32(float(torch.linalg.vector_norm(adjoint.detach()))))
    if 0.0 == eps:
        return True
    # `isnormal`, which is finite and not a subnormal.
    return not (math.isfinite(eps) and abs(eps) >= float(np.finfo(np.float32).tiny))


#: The two iterations that take a term's transform, and so the only two BART
#: lets an extending term reach.  `italgo_choose` sends every one of the three
#: to the alternating directions.
_TAKES_A_TRANSFORM = ("admm", "pridu")

#: What `opt_reg_configure` falls back on, and what a term that does not read
#: these answers with.
_SHARED_OPTIONS = (8, "dau2", 1)


#: The pairs ``pics`` takes once for the whole set, and BART's own values for
#: them (`opt_reg_init`, optreg.c:309-313).  Only the terms that add unknowns
#: read them.
_SHARED_PAIRS: dict[str, tuple[float, float]] = {
    "alpha": (1.0, 3.0**0.5),
    "gamma": (1.0, 1.0),
}


def _shared_pairs(terms) -> tuple[tuple[float, float], tuple[float, float]]:
    """``--alpha`` and ``--gamma`` for the whole set.

    They live on ``struct opt_reg_s`` rather than on a term, which is why
    ``pics`` has a single ``--alpha``; two terms disagreeing is refused rather
    than one of them silently winning, as it is for the block size.
    """
    resolved = dict(_SHARED_PAIRS)
    asked: dict[str, tuple[float, float]] = {}
    for term in terms:
        for name, value in term._settings().items():
            if name not in resolved:
                continue
            if asked.setdefault(name, value) != value:
                raise ValueError(
                    f"{name} is one pair for the whole set -- `pics` has a single "
                    f"--{name} -- and these terms ask for {asked[name]!r} and {value!r}; "
                    "give them the same, or solve for them separately"
                )
            resolved[name] = value
    return resolved["alpha"], resolved["gamma"]


def _extending(terms) -> bool:
    """Whether any of ``terms`` adds unknowns to the optimization variable."""
    return any(getattr(term, "_extends", False) for term in terms)


def _in_library(solver, y, A, x0):
    """The library's loop, for a solver holding a term that adds unknowns.

    There is no written-out step for one -- the iteration here walks the image,
    and this one walks the image and the fields behind it -- so `__call__`
    comes here instead.  The library's loop records nothing, so a tracked ``y``
    is refused rather than answered with a tensor that has quietly lost its
    graph.
    """
    from bartorch.linop.base import _tracking

    # A prior BART cannot be given is the more basic problem, and `in_library`
    # is where that is said; this speaks only for a solve BART can run.
    if not solver._foreign and _tracking(y):
        raise RuntimeError(
            f"{type(solver).__name__} cannot be differentiated through with a term that "
            "adds unknowns to the optimization: the solve runs inside the library, which "
            "is where BART lays that larger vector out, and the library's loop records "
            "nothing.  Detach the data, or regularize with a term that walks the image "
            "alone -- priors.TotalVariation is the one nearest to these"
        )
    return solver.in_library(y, A, x0)


def _shared_options(terms) -> tuple[int, str, int]:
    """The one block size, wavelet family and shift mode for the whole set.

    ``opt_reg_configure`` takes one of each and hands them to whichever terms
    read them, which is how ``pics`` has a single ``-b`` and a single ``-w``.
    A term that reads none answers with the defaults, so what is looked for is
    the terms that said something, and two of those disagreeing is refused
    rather than silently resolved.
    """
    asked = {term._options() for term in terms if term._options() != _SHARED_OPTIONS}
    if not asked:
        return _SHARED_OPTIONS
    if 1 < len(asked):
        raise ValueError(
            "BART configures a set of terms with one block size, one wavelet family and "
            f"one shift mode -- `pics` has a single -b and a single -w -- and these ask for "
            f"{sorted(asked)}; give them the same, or solve for them separately"
        )
    return next(iter(asked))


class _Solver:
    """Base of the solvers that run BART's ``lsqr2``."""

    #: The name ``bartorch_solve`` selects the iteration by.
    _algorithm = ""

    def __init__(self, regularizers: Regularizers, maxiter: int, cclambda: float, precond=None):
        self.regularizers = _as_terms(regularizers)
        if _extending(self.regularizers) and self._algorithm not in _TAKES_A_TRANSFORM:
            raise TypeError(
                f"{type(self).__name__} cannot take a term that adds unknowns to the "
                "optimization: total generalized variation and the two infimal convolutions "
                "split into several penalties over the enlarged variable, and only the "
                "alternating-direction and primal-dual iterations are given a term's "
                "transform at all; optim.ADMM takes them"
            )
        self.maxiter = int(maxiter)
        self.cclambda = float(cclambda)
        self.precond = precond

    @property
    def _foreign(self) -> list:
        """The terms BART could not have been given: a ``deepinv`` prior or a
        denoiser standing where one of its own would."""
        return [t for t in self.regularizers if not isinstance(t, Regularizer)]

    def _settings(self) -> dict:
        return {}

    # --- as a network ------------------------------------------------------

    def _pieces(self, image_shape: tuple[int, ...]):
        """The iteration, the prior and the parameters a loop here drives.

        What :meth:`__call__` assembles, minus everything that needs the
        encoding: a network is handed one per call rather than built around
        one.
        """
        raise TypeError(
            f"{type(self).__name__} runs inside the library and has no iteration written "
            "out here, so there is nothing to unroll; the proximal solvers have one"
        )

    def unrolled(self, image_shape, *, trainable=(), **kwargs):
        """This solver as a network of ``maxiter`` steps, trained end to end.

        The steps are BART's, and the thing standing where a term goes is
        whatever was handed over -- so an unrolled network here is the
        library's iteration with a denoiser in the threshold's place, and not
        an architecture that resembles it.

        Parameters
        ----------
        image_shape : tuple of int
            What the terms are configured for.  A network is not built around
            an encoding, so this is the one shape it has to be told.
        trainable : iterable of str, optional
            Parameters to learn, one value per step: ``"stepsize"`` for the
            proximal-gradient solvers, ``"rho"`` for the alternating
            directions, ``"sigma"`` and ``"tau"`` for the primal-dual.  The
            rest stay the numbers they were given.
        **kwargs
            Passed to ``deepinv.optim.BaseOptim``.  ``custom_init`` defaults
            to starting at zero, where BART starts, rather than at ``A^H y``
            where ``deepinv`` starts an optimizer; passing ``None`` restores
            that other start.

        Returns
        -------
        deepinv.optim.BaseOptim
            A ``torch.nn.Module`` taking ``(y, physics)``, where ``physics``
            is :func:`bartorch.to_deepinv` of the encoding -- or the encoding
            itself, which answers to the same names.

        Notes
        -----
        A learned parameter is a tensor, and an iteration with one in it works
        its scalars out in single precision throughout rather than in a double
        rounded at the end.  So a network does not answer with the library's
        bits, and could not: the numbers in it are no longer the library's.
        With nothing trainable it still does.

        Examples
        --------
        >>> net = optim.FISTA(denoiser, maxiter=10, step=0.9).unrolled(
        ...     (1, 256, 256), trainable=["stepsize"]
        ... )
        >>> torch.optim.Adam(net.parameters(), lr=1e-3)
        >>> net(kspace[None], bartorch.to_deepinv(A))
        """
        return self._network(image_shape, deq=False, trainable=trainable, **kwargs)

    def fixed_point(self, image_shape, *, trainable=(), **kwargs):
        """This solver as a deep-equilibrium model: the step's fixed point.

        The same step as :meth:`unrolled`, run to convergence rather than a
        set number of times, and differentiated through the fixed point rather
        than through the run.  ``maxiter`` is then a cap on the search.

        A step has to be the same map every time for its fixed point to mean
        anything, which is what :class:`FISTA` refuses over: its momentum
        depends on the iteration number.

        Parameters and returns are :meth:`unrolled`'s.
        """
        return self._network(image_shape, deq=True, trainable=trainable, **kwargs)

    def _network(self, image_shape, *, deq: bool, trainable, **kwargs):
        from deepinv.optim import BaseOptim

        from bartorch.optim._iterators import NormalEquations

        if _extending(self.regularizers):
            raise TypeError(
                "a term that adds unknowns to the optimization has no step written out "
                "here: the iterations written out in Python walk the image, and this "
                "one walks the image and the supporting variables behind it.  It solves -- "
                "inside the library, which is where BART lays that vector out -- but it "
                "does not unroll"
            )

        iteration, prior, params = self._pieces(tuple(image_shape))

        trainable = list(trainable)
        unknown = [name for name in trainable if name not in params]
        if unknown:
            raise ValueError(
                f"{type(self).__name__} has no parameter called {unknown[0]!r} to learn; "
                f"it takes {sorted(k for k, v in params.items() if isinstance(v, float))}"
            )

        shape = tuple(image_shape)
        kwargs.setdefault("custom_init", lambda data, physics: _zeros(data, shape))

        network = BaseOptim(
            iteration,
            params_algo=params,
            data_fidelity=NormalEquations(self.cclambda),
            prior=prior,
            max_iter=self.maxiter,
            unfold=not deq,
            DEQ=deq,
            trainable_params=trainable,
            **kwargs,
        )

        if hasattr(iteration, "finish") and "get_output" not in kwargs:
            # `italgo_config` leaves `last` false, so BART thresholds once more
            # after the loop -- which for a network is its final layer, and has
            # to read the parameters as they are now rather than as they were
            # when this was built.
            last = max(self.maxiter - 1, 0)
            network.get_output = lambda X: iteration.finish(
                X["est"][0], prior, network.update_params_fn(last), self.maxiter
            )
        return network

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve for the image given data ``y`` and encoding ``A``.

        Parameters
        ----------
        y : torch.Tensor
            Data of ``A.oshape``.
        A : LinearOperator
            The encoding.  A BART-backed operator is applied without leaving
            the library; a Python-defined one is called back once per
            application.
        x0 : torch.Tensor, optional
            Warm start of ``A.ishape``; without one the iteration starts at
            zero.

        Returns
        -------
        torch.Tensor
            Complex64 solution of ``A.ishape``.
        """
        return self.in_library(y, A, x0)

    def in_library(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve with BART's own loop, without crossing back into Python.

        This is what :meth:`__call__` does for every solver whose iteration is
        the library's.  Where the iteration has been written out in Python --
        so that it can be unrolled into a network or driven to a fixed point --
        :meth:`__call__` runs that one instead, and this stays as the reference
        it is held against: the two answer with the same bits, which is what
        the suite checks.

        A solver holding a ``deepinv`` prior has no library route at all: BART
        has no way to be handed a denoiser, and this says so rather than
        substituting something else.

        Nothing here is recorded for autograd, whichever solver it is: the
        loop is the library's and there is no graph to be had from it.  That
        is what :meth:`__call__` is for.
        """
        if self._foreign:
            raise ValueError(
                f"{self._foreign[0]!r} is not a term BART can be given, so there is no "
                "library route for this solve; the iteration written here is the one that "
                "takes it"
            )
        return _solve(
            A,
            y,
            x0,
            self._algorithm,
            self.regularizers,
            maxiter=self.maxiter,
            cclambda=self.cclambda,
            precond=self.precond,
            **self._settings(),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.regularizers!r}, maxiter={self.maxiter})"


class CG(_Solver):
    """Conjugate gradients for a least-squares problem with quadratic penalties.

    Without terms it is ``min ||A x - y||^2 + lambda_ ||x||^2``, which is what
    ``pics`` runs with no regularizer or with ``-r`` alone.  With terms it is

    ``min ||A x - y||^2 + lambda_ ||x||^2 + sum_i w_i ||G_i x - b_i||^2``

    which is still a least-squares problem and so still this iteration.

    Parameters
    ----------
    lambda_ : float
        Tikhonov weight on the image itself (``pics -r``).  BART adds it to
        the normal operator, which is what makes this one match the tool.
    terms : Tikhonov or iterable of Tikhonov, optional
        Quadratic penalties with an operator, a bias, or both.  See
        :class:`Tikhonov`.
    maxiter : int
    tol : float
        Stop once the residual of the normal equations is at most
        ``tol * ||A^H y||``.  Zero, BART's default, runs every iteration.
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    precond : LinearOperator, optional
        Left preconditioner, ``lsqr2_create``'s ``precond_op``: chained onto
        the normal operator and onto the adjoint, so the iteration sees
        ``M(A^H A + lambda) x = M A^H y``.  Must be positive definite --
        BART composes it without symmetrizing.  ``pics`` passes NULL, so
        nothing on the command line has used it.

    Notes
    -----
    BART's conjugate gradients takes one weight and nothing else:
    ``iter2_conjgrad`` asserts that it is handed no regularizing operators and
    no biases, and ``lsqr2_create`` builds ``A^H A + lambda I``.  So the terms
    are not passed to it -- they are built into the operator it is given, by
    stacking them under the encoding.

    Examples
    --------
    >>> CG(maxiter=30)(kspace, A)
    >>> CG(terms=Tikhonov(0.1, bias=prior))(kspace, A)
    >>> CG(terms=[Tikhonov(0.1, operator=G), Tikhonov(0.01)])(kspace, A)
    """

    _algorithm = "cg"

    def __init__(
        self,
        lambda_: float = 0.0,
        *,
        terms: Terms = None,
        maxiter: int = 30,
        tol: float = 0.0,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(L2(lambda_) if lambda_ else None, maxiter, cclambda, precond)
        self.lambda_ = float(lambda_)
        self.terms = _as_quadratics(terms)
        self.tol = float(tol)

    def _settings(self) -> dict:
        return {"cg_tol": self.tol}

    def __call__(
        self,
        y: torch.Tensor,
        A,
        x0: torch.Tensor | None = None,
        *,
        steps: list | None = None,
    ) -> torch.Tensor:
        """Solve, and with ``steps`` say how many iterations it took.

        The count is what an alternating-direction solver budgets by, and the
        only way to see it from outside the library.

        When ``y`` carries a gradient the solve is recorded: the forward pass
        is the same iteration and the same bits, and the backward pass is
        another solve with the same operator, as
        :mod:`bartorch.optim.autograd` describes.  That is what lets a solve
        stand inside an unrolled network -- the data-consistency layer of a
        MoDL, say -- rather than only at the end of one.
        """
        from bartorch.linop.base import _tracking

        if self.terms:
            A, y = _stacked(A, y, self.terms)

        def forward(data: torch.Tensor) -> torch.Tensor:
            return _solve(
                A,
                data,
                x0,
                self._algorithm,
                self.regularizers,
                maxiter=self.maxiter,
                cclambda=self.cclambda,
                precond=self.precond,
                steps=steps,
                **self._settings(),
            )

        if not _tracking(y):
            return forward(y)

        from bartorch.optim.autograd import apply_solve

        return apply_solve(y, forward, lambda g: A.forward(self._inverse(A, g)))

    def _inverse(self, A, g: torch.Tensor) -> torch.Tensor:
        """``N^-1 g``, driven from the right-hand side rather than from data.

        ``_WithNormal(Identity, A.gram())`` is an operator whose adjoint is the
        identity and whose normal is ``A^H A``, so conjugate gradients on it
        solves ``N w = g`` instead of ``N w = A^H g``.  The same terms and the
        same weight go in, so it is the same ``N`` the forward pass inverted;
        and an encoding built with ``toeplitz=True`` keeps its point-spread
        convolution through :meth:`~bartorch.linop.LinearOperator.gram`, so the
        backward pass costs what the forward one does.
        """
        from bartorch.linop import Identity
        from bartorch.linop.base import _WithNormal

        shape = A.ishape
        return _solve(
            _WithNormal(Identity(shape), A.gram()),
            g,
            None,
            self._algorithm,
            self.regularizers,
            maxiter=self.maxiter,
            cclambda=self.cclambda,
            precond=self.precond,
            **self._settings(),
        )

    def in_library(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """The same as :meth:`__call__`: conjugate gradients stays BART's.

        It is the one solver here with no iteration of its own, because it is
        the one whose step is nothing but the normal operator -- there is no
        proximal operator to unroll around.
        """
        return self(y, A, x0)

    def __repr__(self) -> str:
        terms = f", terms={self.terms!r}" if self.terms else ""
        return f"CG(lambda_={self.lambda_}{terms}, maxiter={self.maxiter}, tol={self.tol})"


class IST(_Solver):
    """Iterative soft thresholding (``pics --ist``).

    Parameters
    ----------
    regularizers : Regularizer or iterable of Regularizer, optional
        Terms from :mod:`bartorch.priors`.
    maxiter : int
    step : float
        Step size (``pics -s``); 0.95 is what ``pics`` uses when none is given.
    eigen : bool
        Scale the step by the largest eigenvalue of the normal operator,
        estimated with 30 power iterations (``pics -e``).
    hogwild : bool
        BART's ``hogwild`` setting (``pics -H``).
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    precond : LinearOperator, optional
        Left preconditioner, ``lsqr2_create``'s ``precond_op``: chained onto
        the normal operator and onto the adjoint, so the iteration sees
        ``M(A^H A + lambda) x = M A^H y``.  Must be positive definite --
        BART composes it without symmetrizing.  ``pics`` passes NULL, so
        nothing on the command line has used it.
    """

    _algorithm = "ist"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        step: float = 0.95,
        eigen: bool = False,
        hogwild: bool = False,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        if 1 != len(self.regularizers):
            # `iter2_ist` and `iter2_fista` assert one, and an assertion in
            # the library takes the process rather than coming back as an
            # error.  ADMM is the iteration that splits several terms apart.
            raise ValueError(
                f"{type(self).__name__} takes exactly one term, not "
                f"{len(self.regularizers)}; ADMM is the one that splits several apart"
            )
        if hogwild and "ist" == self._algorithm:
            # `iter2_ist` asserts it off -- "Let's see whether somebody uses
            # it..." -- and an assertion in the library takes the process,
            # rather than coming back as an error.
            raise ValueError(
                "BART's iterative soft thresholding refuses hogwild; FISTA is the one "
                "that decays its step"
            )
        self.step = float(step)
        self.eigen = bool(eigen)
        self.hogwild = bool(hogwild)

    def _settings(self) -> dict:
        return {"step": self.step, "eigen": self.eigen, "hogwild": self.hogwild}

    def _iteration(self):
        from bartorch.optim._iterators import ISTIteration

        return ISTIteration()

    def _stepsize(self, divisor: float) -> float:
        """``ist`` is handed ``conf->super.alpha * conf->step / maxeigen``.

        Two floats multiplied and then divided by a double, which the compiler
        works out in double and the parameter rounds back to a float.  Written
        out because rounding it anywhere else is a different step.
        """
        return float(np.float32(float(np.float32(self.step)) / divisor))

    def _parameters(self) -> dict:
        return {}

    def _pieces(self, image_shape: tuple[int, ...]):
        from bartorch.optim._iterators import TermPrior

        if self.eigen:
            raise ValueError(
                "eigen=True divides the step by a power iteration over the encoding, and a "
                "network is handed one per call rather than built around one; give the step "
                "itself, or learn it with trainable=['stepsize']"
            )
        params = {
            "maxiter": self.maxiter,
            "stepsize": self._stepsize(1.0),
            "hogwild": self.hogwild,
            **self._parameters(),
        }
        return self._iteration(), TermPrior(self.regularizers[0], image_shape), params

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve, by the iteration written out in Python.

        The loop is here rather than in the library, so that the same solver
        can be unrolled into a network or driven to a fixed point.  The step
        is BART's, held against the library's own to the bit.
        """
        from bartorch import to_deepinv
        from bartorch.optim._iterators import NormalEquations, TermPrior

        iteration = self._iteration()

        op, y, x = _start(A, y, x0, self.regularizers)
        physics = to_deepinv(A)
        fidelity = NormalEquations(self.cclambda)
        if _empty(fidelity._adjoint_data(y, physics)):
            return x

        divisor = maxeigen(A, cclambda=self.cclambda) if self.eigen else 1.0
        prior = TermPrior(self.regularizers[0], op.ishape)
        params = {
            "maxiter": self.maxiter,
            "stepsize": self._stepsize(divisor),
            "hogwild": self.hogwild,
            **self._parameters(),
        }

        state = {"est": (x, x)}
        for _ in range(self.maxiter):
            state = iteration.forward(state, fidelity, prior, params, y, physics)
        return iteration.finish(state["est"][0], prior, params, self.maxiter)


class FISTA(IST):
    """Fast iterative soft thresholding (``pics --fista``).

    Parameters
    ----------
    regularizers : Regularizer or iterable of Regularizer, optional
        Terms from :mod:`bartorch.priors`.
    maxiter : int
    step : float
        Step size (``pics -s``); 0.95 is what ``pics`` uses when none is given.
    eigen : bool
        Scale the step by the largest eigenvalue of the normal operator,
        estimated with 30 power iterations (``pics -e``).
    hogwild : bool
        BART's ``hogwild`` setting (``pics -H``).
    pqr : tuple of float, optional
        Acceleration parameters ``(p, q, r)`` (``pics --fista_pqr``); ``None``
        keeps BART's.
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    precond : LinearOperator, optional
        Left preconditioner, ``lsqr2_create``'s ``precond_op``: chained onto
        the normal operator and onto the adjoint, so the iteration sees
        ``M(A^H A + lambda) x = M A^H y``.  Must be positive definite --
        BART composes it without symmetrizing.  ``pics`` passes NULL, so
        nothing on the command line has used it.
    """

    _algorithm = "fista"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        step: float = 0.95,
        eigen: bool = False,
        hogwild: bool = False,
        pqr: tuple[float, float, float] | None = None,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(
            regularizers,
            maxiter=maxiter,
            step=step,
            eigen=eigen,
            hogwild=hogwild,
            precond=precond,
            cclambda=cclambda,
        )
        self.pqr = None if pqr is None else tuple(float(v) for v in pqr)

    def _settings(self) -> dict:
        return {**super()._settings(), "pqr": self.pqr}

    def _iteration(self):
        from bartorch.optim._iterators import FISTAIteration

        return FISTAIteration()

    def _stepsize(self, divisor: float) -> float:
        """``fista`` is handed ``conf->step / maxeigen`` -- ``alpha`` goes to
        it separately, as the scaling on the threshold, and is one."""
        return float(np.float32(float(np.float32(self.step)) / divisor))

    def _parameters(self) -> dict:
        return {} if self.pqr is None else {"pqr": self.pqr}

    def fixed_point(self, image_shape, *, trainable=(), **kwargs):
        """Refused: the ravine step is not the same map twice.

        ``t <- (p + sqrt(q + r t^2)) / 2`` carries the iteration number into
        the momentum, so the step has no fixed point to find.  Iterative soft
        thresholding is the same iteration without it, and does.
        """
        raise TypeError(
            "FISTA's momentum depends on the iteration number, so its step is a different "
            "map every time and has no fixed point; optim.IST is this iteration without "
            "the ravine, and fixed_point() takes that"
        )


class ADMM(_Solver):
    """Alternating direction method of multipliers (``pics --admm``).

    Parameters
    ----------
    regularizers : Regularizer or iterable of Regularizer, optional
        Terms from :mod:`bartorch.priors`.  Terms that add unknowns to the
        optimization -- total generalized variation and the two infimal
        convolutions -- are taken here and solved inside the library.
    maxiter : int
        A budget on conjugate-gradient iterations across the whole run, not a
        count of outer steps: ``admm`` breaks when ``nr_invokes > maxiter``.
        Thirty with ten inner iterations is about five outer steps.
    rho : float
        Penalty parameter (``pics -u``); 0.5 is BART's default.
    cg_maxiter : int
        Conjugate-gradient iterations per step (``pics -C``); 10 is BART's
        default.
    hogwild : bool
        BART's ``hogwild`` setting (``pics -H``), which doubles ``rho`` after
        ten steps, then twenty, then forty.  Not combinable with
        ``dynamic_rho``, which BART asserts against.
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    biases : sequence of tensor, optional
        The ``b_j`` of ``f_j(G_j x - b_j)``, one per term, each of its term's
        transformed shape.
    dynamic_rho : bool
        Move ``rho`` with the residuals (``pics --admm_dynamic_rho``): up by
        ``tau`` when the primal residual leads, down when the dual does.  The
        dual variables are rescaled to match, so the split stays where it was.
    dynamic_tau : bool
        Choose ``tau`` from the residuals too (``pics --admm_dynamic_tau``),
        as ``sqrt(r / s)`` clipped to ``[1 / tau_max, tau_max]``.  Together
        with ``dynamic_rho`` and ``relative_norm`` this is the residual
        balancing of Wohlberg (2017).
    relative_norm : bool
        Compare the residuals to their scalings rather than to each other
        (``pics --admm_relative_norm``).
    fast : bool
        Skip the residuals entirely, and with them the stopping test.
    alpha : float
        Over-relaxation; BART's default of 1.6 is what ``pics`` runs.  Out of
        reach of :meth:`in_library`, which ``italgo_config`` gives no way to
        set.
    mu : float
        How far the residuals must part before ``dynamic_rho`` moves ``rho``.
        Out of reach of :meth:`in_library`.
    tau_max : float
        The clip on ``tau``.  Out of reach of :meth:`in_library`.
    abstol, reltol : float
        Boyd's absolute and relative tolerances, which stop the iteration when
        both residuals are inside them.  ``italgo_config`` sets both to zero,
        whatever ``iter_admm_defaults`` says, so the budget is what stops
        ``pics``; these are out of reach of :meth:`in_library`.
    cg_maxiter_first : int, optional
        A separate budget for the first step's inner solve, where there is no
        warm start to build on.  Not BART's, so out of reach of
        :meth:`in_library` as well.
    precond : LinearOperator, optional
        Left preconditioner, ``lsqr2_create``'s ``precond_op``: chained onto
        the normal operator and onto the adjoint, so the iteration sees
        ``M(A^H A + lambda) x = M A^H y``.  Must be positive definite --
        BART composes it without symmetrizing.  ``pics`` passes NULL, so
        nothing on the command line has used it.
    """

    _algorithm = "admm"

    #: What `italgo_config` gives no way to set, so a solve that runs inside
    #: the library cannot honour it.  Each is the attribute and the value it
    #: has when nothing was asked for.
    _beyond_the_tool = {
        "alpha": 1.6,
        "mu": 3.0,
        "tau_max": 20.0,
        "abstol": 0.0,
        "reltol": 0.0,
        "cg_maxiter_first": None,
        "biases": None,
    }

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        rho: float = 0.5,
        cg_maxiter: int = 10,
        hogwild: bool = False,
        cclambda: float = 0.0,
        biases: Sequence[torch.Tensor] | None = None,
        dynamic_rho: bool = False,
        dynamic_tau: bool = False,
        relative_norm: bool = False,
        fast: bool = False,
        alpha: float = 1.6,
        mu: float = 3.0,
        tau_max: float = 20.0,
        abstol: float = 0.0,
        reltol: float = 0.0,
        cg_maxiter_first: int | None = None,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        self.rho = float(rho)
        self.cg_maxiter = int(cg_maxiter)
        self.hogwild = bool(hogwild)
        self.dynamic_rho = bool(dynamic_rho)
        self.dynamic_tau = bool(dynamic_tau)
        self.relative_norm = bool(relative_norm)
        self.fast = bool(fast)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.tau_max = float(tau_max)
        self.abstol = float(abstol)
        self.reltol = float(reltol)
        self.cg_maxiter_first = None if cg_maxiter_first is None else int(cg_maxiter_first)
        self.biases = None if biases is None else list(biases)

        if self.hogwild and self.dynamic_rho:
            # `admm` asserts the two apart, and an assertion is the process.
            raise ValueError("BART's ADMM takes hogwild or a dynamic rho, not both")
        if self.fast and self.dynamic_rho:
            # `admm` asserts this one apart too: there are no residuals in
            # fast mode, and a dynamic rho is a move on the residuals.
            raise ValueError("a dynamic rho needs the residuals, which fast mode does not compute")
        if self.biases is not None and len(self.biases) != len(self.regularizers):
            raise ValueError("one bias per term, or none at all")

    def _settings(self) -> dict:
        return {
            "rho": self.rho,
            "cg_maxiter": self.cg_maxiter,
            "hogwild": self.hogwild,
            "dynamic_rho": self.dynamic_rho,
            "dynamic_tau": self.dynamic_tau,
            "relative_norm": self.relative_norm,
            "fast": self.fast,
        }

    def in_library(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """BART's own loop, which cannot be given everything this solver takes.

        ``italgo_config`` builds the configuration ``pics`` builds and hands
        it over; the over-relaxation, ``mu``, ``tau_max`` and the two
        tolerances are not among the things it sets, and neither is a bias or
        riesling's first-step budget.  Asking for one of those and then for
        BART's loop is refused rather than quietly dropped.
        """
        asked = [
            name
            for name, default in self._beyond_the_tool.items()
            if getattr(self, name) != default
        ]
        if asked:
            raise ValueError(
                f"{', '.join(sorted(asked))} cannot be set on BART's own loop -- "
                "`italgo_config` has no way to pass them; the iteration written here does"
            )
        return super().in_library(y, A, x0)

    def fixed_point(self, image_shape, *, trainable=(), **kwargs):
        """Refused: the fixed point of an alternating-direction step is not in ``x``.

        The step moves ``(x, z, u)`` together, and its ``x`` depends on the
        previous ``x`` only as the warm start of the inner solve -- which
        carries no gradient, because what is differentiated is the linear
        system and not the walk towards it.  A deep-equilibrium model built on
        ``x`` alone would therefore be differentiating a map that does not
        depend on its argument.

        :class:`IST` and :class:`PRIDU` have fixed points in the iterate and
        take this.
        """
        raise TypeError(
            "an alternating-direction step's fixed point is in (x, z, u) rather than in the "
            "image, and its x-update depends on the previous image only through a warm start "
            "that carries no gradient; optim.IST and optim.PRIDU take fixed_point()"
        )

    def _pieces(self, image_shape: tuple[int, ...]):
        from bartorch.optim._iterators import ADMMIteration

        params = {
            "maxiter": self.maxiter,
            "cg_maxiter": self.cg_maxiter,
            "cg_maxiter_first": self.cg_maxiter_first,
            "rho": self.rho,
            "hogwild": self.hogwild,
            "cclambda": self.cclambda,
            "dynamic_rho": self.dynamic_rho,
            "dynamic_tau": self.dynamic_tau,
            "relative_norm": self.relative_norm,
            "fast": self.fast,
            "alpha": self.alpha,
            "mu": self.mu,
            "tau_max": self.tau_max,
            "abstol": self.abstol,
            "reltol": self.reltol,
        }
        iteration = ADMMIteration(self.regularizers, image_shape, biases=self.biases)
        # ADMM asks the terms for their proximal operators itself, so the
        # prior slot a `deepinv` optimizer would fill is empty here.
        return iteration, None, params

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve, by the iteration written out in Python.

        The loop is here rather than in the library, so that the same solver
        can be unrolled into a network or driven to a fixed point.  The step
        is BART's -- every operator in it is, and the arithmetic is held
        against the library's own to the bit -- and so is when it stops.

        """
        from bartorch import to_deepinv
        from bartorch.optim._iterators import NormalEquations

        if _extending(self.regularizers):
            # The step written out here walks the image; this one walks the
            # image and the supporting variables behind it, and BART is where
            # that vector is laid out.  Same answer, the library's loop.
            return _in_library(self, y, A, x0)

        op, y, x = _start(A, y, x0, self.regularizers)

        iteration, _, params = self._pieces(op.ishape)
        iteration.restart()

        state = {"est": (x, x)}
        for _ in range(self.maxiter):
            state = iteration.forward(
                state,
                NormalEquations(),
                None,
                params,
                y,
                to_deepinv(A),
            )
            if state["done"]:
                break
        return state["est"][0]


class PRIDU(_Solver):
    """Primal-dual iteration (``pics --pridu``).

    Parameters
    ----------
    regularizers : Regularizer or iterable of Regularizer, optional
        Terms from :mod:`bartorch.priors`.  Terms that add unknowns to the
        optimization are taken here and solved inside the library, as
        :class:`ADMM` takes them.
    maxiter : int
    step : float
        Step size (``pics -s``); 0.95 is what ``pics`` uses when none is given.
    sigma_tau_ratio : float
        Ratio of the dual to the primal step: ``sigma = sqrt(step) * ratio``,
        ``tau = sqrt(step) / ratio``.  ``pics`` sets it to the scaling it
        divided the data by, so pass :func:`data_scaling`'s value to match
        the tool.
    adaptive_step : bool
        Adapt the steps during the iteration (``pics --adaptive_stepsize``).
    eigen : bool
        Scale the step by the largest eigenvalue of the normal operator,
        estimated with 30 power iterations (``pics -e``).
    hogwild : bool
        Decay the steps by a factor of 0.95 (``pics -H``).
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    precond : LinearOperator, optional
        Left preconditioner, ``lsqr2_create``'s ``precond_op``: chained onto
        the normal operator and onto the adjoint, so the iteration sees
        ``M(A^H A + lambda) x = M A^H y``.  Must be positive definite --
        BART composes it without symmetrizing.  ``pics`` passes NULL, so
        nothing on the command line has used it.
    """

    _algorithm = "pridu"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        step: float = 0.95,
        sigma_tau_ratio: float = 1.0,
        adaptive_step: bool = False,
        eigen: bool = False,
        hogwild: bool = False,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        self.step = float(step)
        self.sigma_tau_ratio = float(sigma_tau_ratio)
        self.adaptive_step = bool(adaptive_step)
        self.eigen = bool(eigen)
        self.hogwild = bool(hogwild)

    def _settings(self) -> dict:
        return {
            "step": self.step,
            "sigma_tau_ratio": self.sigma_tau_ratio,
            "adaptive_step": self.adaptive_step,
            "eigen": self.eigen,
            "hogwild": self.hogwild,
        }

    def _split(self, image_shape: tuple[int, ...]):
        """The primal term and the dual ones, as ``iter2_chambolle_pock``
        splits them: the first term, if its transform is the identity, becomes
        the primal proximal step and the rest become duals.  Otherwise every
        term is a dual and the primal step is ``prox_zero_create``'s, which is
        to leave the image alone."""
        terms = list(self.regularizers)
        if terms and terms[0].transform_is_identity(image_shape):
            return terms[0], terms[1:]
        return None, terms

    def _steps(self, divisor: float) -> tuple[float, float, float]:
        """``sigma``, ``tau`` and the ratio between them, each rounded where
        ``iter2_chambolle_pock`` rounds it."""
        root = float(np.float32(math.sqrt(self.step)))
        ratio = float(np.float32(self.sigma_tau_ratio))
        return (
            float(np.float32(float(np.float32(root * ratio)) / divisor)),
            float(np.float32(float(np.float32(root / ratio)) / divisor)),
            ratio,
        )

    def _pieces(self, image_shape: tuple[int, ...], divisor: float | None = None):
        from bartorch.optim._iterators import PRIDUIteration

        if divisor is None and self.eigen:
            raise ValueError(
                "eigen=True divides the steps by a power iteration over the encoding and the "
                "dual terms' transforms, and a network is handed an encoding per call rather "
                "than built around one; give the step itself, or learn the two with "
                "trainable=['sigma', 'tau']"
            )
        primal, duals = self._split(image_shape)
        sigma, tau, ratio = self._steps(1.0 if divisor is None else divisor)
        params = {
            "maxiter": self.maxiter,
            "sigma": sigma,
            "tau": tau,
            "sigma_tau_ratio": ratio,
            "theta": 1.0,
            "decay": 0.95 if self.hogwild else 1.0,
            "tol": 1e-4,
            "adaptive_step": self.adaptive_step,
        }
        # The terms are the iteration's own duals, so the prior slot is empty.
        return PRIDUIteration(duals, image_shape, primal=primal), None, params

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve, by the iteration written out in Python.

        The loop is here rather than in the library, so that the same solver
        can be unrolled into a network or driven to a fixed point.  The steps
        are BART's, held against the library's own to the bit.
        """
        from bartorch import to_deepinv
        from bartorch.optim._iterators import NormalEquations

        if _extending(self.regularizers):
            # As for the alternating directions: the enlarged variable is laid
            # out inside the library, so that is where this one is solved.
            return _in_library(self, y, A, x0)

        op, y, x = _start(A, y, x0, self.regularizers)

        # `iter2_chambolle_pock` estimates over the encoding and the dual
        # terms' transforms together, the primal one having been taken out of
        # the list before it looks.
        divisor = 1.0
        if self.eigen:
            _, duals = self._split(op.ishape)
            divisor = math.sqrt(maxeigen(A, duals, cclambda=self.cclambda))

        iteration, _, params = self._pieces(op.ishape, divisor)
        fidelity = NormalEquations(self.cclambda)

        state = {"est": (x, x)}
        for _ in range(self.maxiter):
            state = iteration.forward(state, fidelity, None, params, y, to_deepinv(A))
            if state["done"]:
                break
        return state["est"][0]


class NIHT(_Solver):
    """Normalized iterative hard thresholding.

    Parameters
    ----------
    regularizers : WaveletNIHT or ImageNIHT, or an iterable of them
        The hard-thresholding terms from :mod:`bartorch.priors`.
    maxiter : int
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    precond : LinearOperator, optional
        Left preconditioner, ``lsqr2_create``'s ``precond_op``: chained onto
        the normal operator and onto the adjoint, so the iteration sees
        ``M(A^H A + lambda) x = M A^H y``.  Must be positive definite --
        BART composes it without symmetrizing.  ``pics`` passes NULL, so
        nothing on the command line has used it.
    """

    _algorithm = "niht"

    def __init__(
        self,
        regularizers: Regularizers,
        *,
        maxiter: int = 30,
        cclambda: float = 0.0,
        precond=None,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        for term in self.regularizers:
            if term.kind not in ("H", "N"):
                raise TypeError(f"NIHT takes WaveletNIHT and ImageNIHT terms, not {term!r}")

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Refused: BART's own iteration cannot run against ``lsqr``'s operator.

        ``niht`` applies the normal operator in place -- ``iter_op_call(op, g,
        g)`` at ``iter/niht.c:85`` and ``:212`` -- and the operator ``lsqr2``
        hands it asserts against exactly that, ``args[0] != args[1]`` at
        ``iter/lsqr.c:60``.  Every NIHT solve therefore ends in an assertion,
        ``bart pics -R H`` included, and BART's assertions are ``error()``
        calls that unwind the process from here rather than returning.
        """
        raise NotImplementedError(
            "BART's NIHT cannot run: `niht` applies the normal operator in place "
            "(iter/niht.c:85, :212) and the operator `lsqr2` hands it asserts that it "
            "is not (iter/lsqr.c:60), so every solve ends in an assertion -- `bart pics "
            "-R H` included.  Nothing here can work around it; it needs a BART fix.  "
            "priors.WaveletNIHT and priors.ImageNIHT still reach tools.pics, which catches "
            "the assertion rather than ending the process"
        )
