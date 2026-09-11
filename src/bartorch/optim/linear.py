"""Regularized least squares by the iterations ``pics`` runs.

Each solver hands BART's ``lsqr2`` the encoding, the operators BART built for
the :mod:`bartorch.prox` terms, and its settings, through the same
``italgo_config`` call ``pics`` makes.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand
from bartorch.prox.base import Regularizer, _as_terms
from bartorch.prox.terms import L2

__all__ = ["ADMM", "CG", "EulerMaruyama", "FISTA", "IST", "NIHT", "PRIDU"]

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
) -> torch.Tensor:
    """Run ``bartorch_solve``.  A negative ``step`` or ``rho``, a zero
    ``cg_maxiter`` and ``pqr=None`` keep BART's defaults."""
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
        )
    if code != 0:
        said = lib.bartorch_solve_error(code).decode(errors="replace")
        raise BartError(f"the solve failed: {said}")
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
    """Conjugate gradients for ``min ||A x - y||^2 + lambda_ ||x||^2``.

    What ``pics`` runs with no regularizer, or with ``-r`` alone.

    Parameters
    ----------
    lambda_ : float
        Tikhonov weight (``pics -r``).
    maxiter : int
    tol : float
        Stop once the residual of the normal equations is at most
        ``tol * ||A^H y||``.  Zero, BART's default, runs every iteration.
    cclambda : float
        Weight of an identity added to the normal operator (``pics -q``).
    """

    _algorithm = "cg"

    def __init__(
        self, lambda_: float = 0.0, *, maxiter: int = 30, tol: float = 0.0, cclambda: float = 0.0
    ):
        super().__init__(L2(lambda_) if lambda_ else None, maxiter, cclambda)
        self.lambda_ = float(lambda_)
        self.tol = float(tol)

    def _settings(self) -> dict:
        return {"cg_tol": self.tol}

    def __repr__(self) -> str:
        return f"CG(lambda_={self.lambda_}, maxiter={self.maxiter}, tol={self.tol})"


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
