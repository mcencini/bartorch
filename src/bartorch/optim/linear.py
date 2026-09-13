"""Regularized least squares by the iterations ``pics`` runs.

Each solver hands BART's ``lsqr2`` the encoding, the operators BART built for
the :mod:`bartorch.prox` terms, and its settings, through the same
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
from bartorch.prox.base import Regularizer, _as_terms
from bartorch.prox.terms import L2

__all__ = ["ADMM", "CG", "PRIDU", "EulerMaruyama", "FISTA", "IST", "NIHT", "Tikhonov"]


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

    >>> CG(terms=Tikhonov(0.1, operator=linop.Gradient(A.ishape)))(y, A)
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

    That last part is the point.  A^H A for a Toeplitz encoding is a
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
    sampler_precond=None,
    sampler_precond_diag: float = 0.0,
    sampler_precond_tol: float = 0.0,
    sampler_precond_maxiter: int = 10,
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
    handles = [term.build(op.ishape) for term in terms]
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
    sampler = None if sampler_precond is None else sampler_precond._bart()

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
            _marshal.pointers(handles) if terms else None,
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
            None if sampler is None else sampler._h.ptr,
            float(sampler_precond_diag),
            float(sampler_precond_tol),
            int(sampler_precond_maxiter),
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


def _empty(adjoint: torch.Tensor) -> bool:
    """``checkeps``: BART declines to iterate on data whose adjoint has no
    norm, or whose norm is not a normal number, and leaves the image be."""
    eps = float(np.float32(float(torch.linalg.vector_norm(adjoint))))
    if 0.0 == eps:
        return True
    # `isnormal`, which is finite and not a subnormal.
    return not (math.isfinite(eps) and abs(eps) >= float(np.finfo(np.float32).tiny))


class _Solver:
    """Base of the solvers that run BART's ``lsqr2``."""

    #: The name ``bartorch_solve`` selects the iteration by.
    _algorithm = ""

    def __init__(self, regularizers: Regularizers, maxiter: int, cclambda: float, precond=None):
        self.regularizers = _as_terms(regularizers)
        for term in self.regularizers:
            if getattr(term, "_extends", False):
                raise TypeError(
                    f"{type(term).__name__} adds variables to the optimization, which BART "
                    "configures only for the whole set of terms at once; tools.pics takes it"
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
        the library's.  Where the iteration has been written out in
        :mod:`bartorch.optim.iterators` -- so that it can be unrolled into a
        network or driven to a fixed point -- :meth:`__call__` runs that one
        instead, and this stays as the reference it is held against: the two
        answer with the same bits, which is what the suite checks.

        A solver holding a ``deepinv`` prior has no library route at all: BART
        has no way to be handed a denoiser, and this says so rather than
        substituting something else.
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

    Notes
    -----
    BART's conjugate gradients takes one weight and nothing else:
    ``iter2_conjgrad`` asserts that it is handed no regularizing operators and
    no biases, and ``lsqr2_create`` builds ``A^H A + lambda I``.  So the terms
    are not passed to it -- they are built into the operator it is given, as
    the stack above, which needs nothing of BART that was not already there.

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
        """
        if self.terms:
            A, y = _stacked(A, y, self.terms)

        return _solve(
            A,
            y,
            x0,
            self._algorithm,
            self.regularizers,
            maxiter=self.maxiter,
            cclambda=self.cclambda,
            precond=self.precond,
            steps=steps,
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
        Terms from :mod:`bartorch.prox`.
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
        from bartorch.optim.iterators import ISTIteration

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

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve, by the iteration in :mod:`bartorch.optim.iterators`.

        The loop is here rather than in the library, so that the same solver
        can be unrolled into a network or driven to a fixed point.  The step
        is BART's, held against the library's own to the bit.
        """
        from bartorch import to_deepinv
        from bartorch.optim.iterators import NormalEquations, TermPrior

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
        Terms from :mod:`bartorch.prox`.
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
        from bartorch.optim.iterators import FISTAIteration

        return FISTAIteration()

    def _stepsize(self, divisor: float) -> float:
        """``fista`` is handed ``conf->step / maxeigen`` -- ``alpha`` goes to
        it separately, as the scaling on the threshold, and is one."""
        return float(np.float32(float(np.float32(self.step)) / divisor))

    def _parameters(self) -> dict:
        return {} if self.pqr is None else {"pqr": self.pqr}


class ADMM(_Solver):
    """Alternating direction method of multipliers (``pics --admm``).

    Parameters
    ----------
    regularizers : Regularizer or iterable of Regularizer, optional
        Terms from :mod:`bartorch.prox`.
    maxiter : int
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
        A separate budget for the first step's inner solve.  Not BART's --
        it is riesling's ``iters0``, the one thing that implementation has
        that BART's does not -- so it is out of reach of :meth:`in_library`
        as well.
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

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve, by the iteration in :mod:`bartorch.optim.iterators`.

        The loop is here rather than in the library, so that the same solver
        can be unrolled into a network or driven to a fixed point.  The step
        is BART's -- every operator in it is, and the arithmetic is held
        against the library's own to the bit -- and so is when it stops.

        """
        from bartorch import to_deepinv
        from bartorch.optim.iterators import ADMMIteration, NormalEquations

        op, y, x = _start(A, y, x0, self.regularizers)

        iteration = ADMMIteration(self.regularizers, op.ishape, biases=self.biases)
        iteration.restart()
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
        Terms from :mod:`bartorch.prox`.
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

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve, by the iteration in :mod:`bartorch.optim.iterators`.

        The loop is here rather than in the library, so that the same solver
        can be unrolled into a network or driven to a fixed point.  The steps
        are BART's, held against the library's own to the bit.
        """
        from bartorch import to_deepinv
        from bartorch.optim.iterators import NormalEquations, PRIDUIteration

        op, y, x = _start(A, y, x0, self.regularizers)
        primal, duals = self._split(op.ishape)

        # `iter2_chambolle_pock` estimates over the encoding and the dual
        # terms' transforms together, the primal one having been taken out of
        # the list before it looks.
        divisor = math.sqrt(maxeigen(A, duals, cclambda=self.cclambda)) if self.eigen else 1.0
        root = float(np.float32(math.sqrt(self.step)))
        ratio = float(np.float32(self.sigma_tau_ratio))

        params = {
            "maxiter": self.maxiter,
            "sigma": float(np.float32(float(np.float32(root * ratio)) / divisor)),
            "tau": float(np.float32(float(np.float32(root / ratio)) / divisor)),
            "sigma_tau_ratio": ratio,
            "theta": 1.0,
            "decay": 0.95 if self.hogwild else 1.0,
            "tol": 1e-4,
            "adaptive_step": self.adaptive_step,
        }

        iteration = PRIDUIteration(duals, op.ishape, primal=primal)
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
        The hard-thresholding terms from :mod:`bartorch.prox`.
    maxiter : int
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
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


class EulerMaruyama(_Solver):
    """BART's Euler-Maruyama iteration (``pics --eulermaruyama``).

    Parameters
    ----------
    regularizers : Regularizer or iterable of Regularizer, optional
        Terms from :mod:`bartorch.prox`.
    step : float
        Step size (``pics -s``).  Required: ``pics`` supplies no default for
        this iteration.
    maxiter : int
    eigen : bool
        Scale the step by the largest eigenvalue of the normal operator,
        estimated with 30 power iterations (``pics -e``).
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    """

    _algorithm = "eulermaruyama"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        step: float,
        maxiter: int = 30,
        eigen: bool = False,
        cclambda: float = 0.0,
        precond=None,
        sampler_precond=None,
        sampler_precond_diag: float = 0.0,
        sampler_precond_tol: float = 0.0,
        sampler_precond_maxiter: int = 10,
    ):
        super().__init__(regularizers, maxiter, cclambda, precond)
        self.step = float(step)
        self.eigen = bool(eigen)
        self.sampler_precond = sampler_precond
        self.sampler_precond_diag = float(sampler_precond_diag)
        self.sampler_precond_tol = float(sampler_precond_tol)
        self.sampler_precond_maxiter = int(sampler_precond_maxiter)
        if sampler_precond is not None and 0.0 >= self.sampler_precond_diag:
            raise ValueError(
                "the sampler's preconditioner is used only when its diagonal is "
                "positive; BART reads the diagonal first and leaves the plain "
                "iteration when it is zero"
            )

    def _settings(self) -> dict:
        return {
            "step": self.step,
            "eigen": self.eigen,
            "sampler_precond": self.sampler_precond,
            "sampler_precond_diag": self.sampler_precond_diag,
            "sampler_precond_tol": self.sampler_precond_tol,
            "sampler_precond_maxiter": self.sampler_precond_maxiter,
        }
