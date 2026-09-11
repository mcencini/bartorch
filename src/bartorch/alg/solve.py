"""The solve ``pics`` runs, assembled here.

Nothing in this module is an algorithm.  BART turns a solver flag and a set of
``-R`` strings into an iteration and a list of proximal operators, and hands
them with the encoding to ``lsqr2``; this hands BART the same three things
through the same functions.  An encoding built out of
:mod:`bartorch.linop` and solved here is the tool's own computation, and
``tests/test_solve.py`` holds it against ``bart pics`` exactly.
"""

from __future__ import annotations

import torch

from bartorch import _marshal
from bartorch._lib import library
from bartorch._operator import as_operand
from bartorch.core.graph import BartError, _ensure_ready, _lock, _on_device

__all__ = ["ALGORITHMS", "REGULARIZERS", "solve"]

#: The transform letters BART's own ``-R`` parser knows, from the run of
#: comparisons in ``grecon/optreg.c``.  It is checked here rather than there
#: because BART answers an unrecognised one with ``error()``, and what that
#: leaves behind makes the next call into the library spin at full CPU -- so a
#: typo in a regularizer would end the session, exactly as an unknown option
#: used to.  ``tests/test_solve.py`` holds this against BART's sources.
REGULARIZERS = frozenset(
    "C F G H I L M N P Q R1 R2 S T TF V W".split()
)

#: The iterations BART has, by the name its own flag gives each.  ``None``
#: leaves the choice to BART, which is what the tool does when no solver flag
#: is given: conjugate gradients without a regularizer, FISTA or ADMM with one,
#: depending on what the regularizers need.
ALGORITHMS = ("cg", "ist", "fista", "admm", "pridu", "niht", "eulermaruyama")


def solve(
    A,
    y: torch.Tensor,
    *,
    regularizers: str | list[str] | None = None,
    solver: str | None = None,
    lambda_: float | None = None,
    cclambda: float = 0.0,
    maxiter: int = 30,
    step: float | None = None,
    eigen: bool = False,
    hogwild: bool = False,
    admm_rho: float | None = None,
    cg_maxiter: int | None = None,
    fista: tuple[float, float, float] | None = None,
    llr_block: int = 8,
    wavelet: str = "dau2",
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve ``min ||A x - y||^2 + lambda ||x||^2 + sum g_i(x)`` the way BART does.

    Parameters
    ----------
    A : LinearOperator
        The encoding.  One of BART's own never leaves C for the length of the
        solve; one written in Python is called back into once per application.
    y : torch.Tensor
        The data, of ``A.oshape``.
    regularizers : str or list of str, optional
        The ``-R`` specifications ``pics`` takes, read by BART's own parser --
        ``"W:7:0:0.005"`` is wavelet regularization on the first three axes.
        :func:`bartorch.tools.describe` does not cover these; ``bart pics -Rh``
        is their documentation, and they mean the same thing here.
    solver : {'cg', 'ist', 'fista', 'admm', 'pridu', 'niht', 'eulermaruyama'}, optional
        Which of BART's iterations to run.  ``None`` lets BART choose, as it
        does for the tool.
    lambda_ : float, optional
        The regularizers' weight (``pics -r``).  ``None`` leaves it unset,
        which is what the regularizer specifications read as "take mine".
    cclambda : float
        The weight in the normal equations (``pics -q``).
    maxiter : int
        Iterations (``pics -i``).
    step : float, optional
        Step size (``pics -s``); ``None`` leaves BART its default.
    eigen : bool
        Scale the step by the largest eigenvalue (``pics -e``).
    hogwild : bool
        BART's Hogwild stepping (``pics -H``).
    admm_rho : float, optional
        ADMM penalty (``pics -u``).
    cg_maxiter : int, optional
        Inner conjugate-gradient steps for ADMM (``pics -C``).
    fista : tuple of float, optional
        FISTA's three acceleration parameters (``pics --fista_pqr``).
    llr_block : int
        Block size for locally low-rank regularization (``pics -b``).
    wavelet : str
        Wavelet family for wavelet regularization (``pics --wavelet``).
    x0 : torch.Tensor, optional
        Warm start; without one BART starts where it starts.

    Returns
    -------
    torch.Tensor
        The solution, of ``A.ishape``.

    Examples
    --------
    >>> from bartorch import alg, linop
    >>> A = linop.Sense(maps, (8, 128, 128), traj=traj)
    >>> x = alg.solve(A, kspace, regularizers="W:7:0:0.005", solver="fista", maxiter=50)

    Notes
    -----
    This is the same computation as ``bartorch.tools.pics`` on the same
    operator, and is for when the encoding is assembled rather than named: a
    SENSE operator over k-space kernels, a chain with something of one's own in
    it, a normal operator brought in from outside.  Where the tool can express
    the problem, the tool is the shorter way to say it.
    """
    if solver is not None and solver not in ALGORITHMS:
        raise ValueError(f"solver must be one of {list(ALGORITHMS)} or None, not {solver!r}")

    specs = [] if regularizers is None else (
        [regularizers] if isinstance(regularizers, str) else list(regularizers)
    )
    for spec in specs:
        transform = spec.split(":", 1)[0]
        if transform not in REGULARIZERS:
            raise ValueError(
                f"{transform!r} is not a regularizer BART has; it takes one of "
                f"{sorted(REGULARIZERS)}.  `bart pics -Rh` describes each."
            )

    bart_op = A.as_bart()
    y = as_operand(y, bart_op.oshape, "y")
    if x0 is None:
        x = torch.zeros(bart_op.ishape, dtype=torch.complex64, device=y.device)
    else:
        x = as_operand(x0, bart_op.ishape, "x0").clone()

    _ensure_ready()
    lib = library()
    specs_argv = _marshal.argv(specs)
    p, q, r = fista if fista is not None else (-1.0, -1.0, -1.0)

    with _lock, _on_device(bart_op.device or y.device):
        code = lib.bartorch_solve(
            bart_op._h.ptr,
            None if solver is None else solver.encode(),
            specs_argv if specs else None,
            len(specs),
            float(lambda_) if lambda_ is not None else -1.0,
            float(cclambda),
            int(maxiter),
            float(step) if step is not None else -1.0,
            int(eigen),
            int(hogwild),
            float(admm_rho) if admm_rho is not None else -1.0,
            int(cg_maxiter) if cg_maxiter is not None else 0,
            float(p),
            float(q),
            float(r),
            int(llr_block),
            0,
            wavelet.encode(),
            int(x0 is not None),
            x.data_ptr(),
            y.data_ptr(),
        )
    if code != 0:
        said = lib.bartorch_solve_error(code).decode(errors="replace")
        raise BartError(f"the solve failed: {said}")
    return x
