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
    pqr: tuple[float, float, float] | None = None,
    sigma_tau_ratio: float = 1.0,
    adaptive_step: bool = False,
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
            float(p),
            float(q),
            float(r),
            float(sigma_tau_ratio),
            int(adaptive_step),
            int(x0 is not None),
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


class _Solver:
    """Base of the solvers that run BART's ``lsqr2``."""

    #: The name ``bartorch_solve`` selects the iteration by.
    _algorithm = ""

    def __init__(self, regularizers: Regularizers, maxiter: int, cclambda: float):
        self.regularizers = _as_terms(regularizers)
        for term in self.regularizers:
            if term._extends:
                raise TypeError(
                    f"{type(term).__name__} adds variables to the optimization, which BART "
                    "configures only for the whole set of terms at once; tools.pics takes it"
                )
        self.maxiter = int(maxiter)
        self.cclambda = float(cclambda)

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
        """
        return _solve(
            A,
            y,
            x0,
            self._algorithm,
            self.regularizers,
            maxiter=self.maxiter,
            cclambda=self.cclambda,
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
    ):
        super().__init__(L2(lambda_) if lambda_ else None, maxiter, cclambda)
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
    ):
        super().__init__(regularizers, maxiter, cclambda)
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
    ):
        super().__init__(
            regularizers,
            maxiter=maxiter,
            step=step,
            eigen=eigen,
            hogwild=hogwild,
            cclambda=cclambda,
        )
        self.pqr = None if pqr is None else tuple(float(v) for v in pqr)

    def _settings(self) -> dict:
        return {**super()._settings(), "pqr": self.pqr}


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
        BART's ``hogwild`` setting (``pics -H``).
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    """

    _algorithm = "admm"

    def __init__(
        self,
        regularizers: Regularizers = None,
        *,
        maxiter: int = 30,
        rho: float = 0.5,
        cg_maxiter: int = 10,
        hogwild: bool = False,
        cclambda: float = 0.0,
    ):
        super().__init__(regularizers, maxiter, cclambda)
        self.rho = float(rho)
        self.cg_maxiter = int(cg_maxiter)
        self.hogwild = bool(hogwild)

    def _settings(self) -> dict:
        return {"rho": self.rho, "cg_maxiter": self.cg_maxiter, "hogwild": self.hogwild}

    def __call__(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Solve, by the iteration in :mod:`bartorch.optim.iterators`.

        The loop is here rather than in the library, so that the same solver
        can be unrolled into a network or driven to a fixed point.  The step
        is BART's -- every operator in it is, and the arithmetic is held
        against the library's own to the bit -- and so is when it stops.

        """
        from bartorch import to_deepinv
        from bartorch.optim.iterators import ADMMIteration, NormalEquations

        op = A._bart()
        y = as_operand(y, op.oshape, "y")
        x = (
            torch.zeros(op.ishape, dtype=torch.complex64, device=y.device)
            if x0 is None
            else as_operand(x0, op.ishape, "x0").clone()
        )

        iteration = ADMMIteration(self.regularizers, op.ishape)
        iteration.restart()
        params = {
            "maxiter": self.maxiter,
            "cg_maxiter": self.cg_maxiter,
            "rho": self.rho,
            "hogwild": self.hogwild,
            "cclambda": self.cclambda,
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
    ):
        super().__init__(regularizers, maxiter, cclambda)
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

    def __init__(self, regularizers: Regularizers, *, maxiter: int = 30, cclambda: float = 0.0):
        super().__init__(regularizers, maxiter, cclambda)
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
    ):
        super().__init__(regularizers, maxiter, cclambda)
        self.step = float(step)
        self.eigen = bool(eigen)

    def _settings(self) -> dict:
        return {"step": self.step, "eigen": self.eigen}
