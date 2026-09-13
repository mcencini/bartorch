"""Preconditioning: what BART has, and what it does not.

``conjgrad`` has no preconditioner argument at all.  What BART calls
preconditioning is three unrelated things, and only one of them touches the
linear solvers:

* ``lsqr2_create``'s ``precond_op``, chained onto the normal operator and the
  adjoint -- left preconditioning by composition.  It is plumbed all the way
  through ``sense_recon_create`` and ``pics.c`` passes NULL, so nothing on the
  command line has ever used it.  That is what ``precond=`` is.
* ``pics --precond``, which is not preconditioning at all: it adds a
  ``prox_weighted_leastsquares`` term with the inverse sampling pattern as
  weights and chains it through the model operator.  It is reachable through
  :func:`bartorch.tools.pics`, where it belongs.
* ``eulermaruyama_precond``, a genuinely preconditioned sampler that runs a
  conjugate-gradient solve of its own at every step.  ``pics`` has no flag for
  it; ``sampler_precond=`` is the only way to reach it.
"""

import pytest
import torch

import bartorch.tools as bt
from bartorch import linop, optim, prox


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _ill_conditioned(n: int = 32):
    """A diagonal encoding whose singular values span two and a half decades."""
    scale = torch.logspace(0, -2.5, n, dtype=torch.float32).to(torch.complex64)
    return linop.Diagonal(scale, (n,)), scale


# --- lsqr's preconditioner --------------------------------------------------


def test_a_preconditioner_turns_a_slow_problem_into_a_fast_one():
    A, scale = _ill_conditioned()
    x = _rand(32)
    y = A(x)
    # The exact inverse of the normal operator, which is the best a diagonal
    # preconditioner can be: it makes the iteration converge in one step.
    M = linop.Diagonal((1.0 / scale.abs() ** 2).to(torch.complex64), (32,))
    plain = optim.CG(maxiter=12)(y, A)
    preconditioned = optim.CG(maxiter=3, precond=M)(y, A)
    assert (plain - x).norm() / x.norm() > 0.5
    assert (preconditioned - x).norm() / x.norm() < 1e-5


def test_the_identity_preconditioner_changes_nothing():
    A, _ = _ill_conditioned(16)
    y = A(_rand(16))
    plain = optim.CG(maxiter=8)(y, A)
    identity = optim.CG(maxiter=8, precond=linop.Identity((16,)))(y, A)
    torch.testing.assert_close(identity, plain, rtol=1e-4, atol=1e-5)


def test_no_preconditioner_is_what_every_solve_did_before():
    A, _ = _ill_conditioned(16)
    y = A(_rand(16))
    assert optim.CG(maxiter=8).precond is None
    torch.testing.assert_close(
        optim.CG(maxiter=8, precond=None)(y, A), optim.CG(maxiter=8)(y, A), rtol=0, atol=0
    )


def test_a_preconditioner_of_the_wrong_shape_says_so():
    A, _ = _ill_conditioned(16)
    y = A(_rand(16))
    with pytest.raises(ValueError, match="maps the image to itself"):
        optim.CG(maxiter=4, precond=linop.FFT((8, 8), axes=-1))(y, A)


@pytest.mark.parametrize(
    "make",
    [
        lambda M: optim.CG(maxiter=6, precond=M),
        lambda M: optim.FISTA(prox.L1(0.001), maxiter=6, precond=M),
        lambda M: optim.ADMM(prox.L1(0.001), maxiter=6, precond=M),
        lambda M: optim.PRIDU(prox.L1(0.001), maxiter=6, precond=M),
    ],
    ids=["cg", "fista", "admm", "pridu"],
)
def test_every_least_squares_solver_takes_one(make):
    # lsqr2_create is where the preconditioner enters, and every one of these
    # goes through it, so every one of them can be given one.
    A, scale = _ill_conditioned(16)
    y = A(_rand(16))
    M = linop.Diagonal((1.0 / scale.abs() ** 2).to(torch.complex64), (16,))
    made = make(M).in_library(y, A)
    assert torch.isfinite(made).all()


def test_a_python_defined_preconditioner_is_called_back():
    A, scale = _ill_conditioned(16)
    y = A(_rand(16))
    weight = (1.0 / scale.abs() ** 2).to(torch.complex64)
    seen = []

    def forward(x):
        seen.append(1)
        return weight * x

    M = linop.Callback((16,), (16,), forward, lambda v: weight.conj() * v)
    made = optim.CG(maxiter=4, precond=M)(y, A)
    assert seen, "the preconditioner was never applied"
    assert torch.isfinite(made).all()


# --- the sampler's own ------------------------------------------------------


def test_the_sampler_takes_a_preconditioner_pics_cannot_reach():
    A = linop.FFT((8, 8), axes=-1)
    y = A(_rand(8, 8))
    settings = dict(step=0.1, maxiter=5)
    plain = optim.EulerMaruyama(prox.L2(0.01), **settings)(y, A)
    preconditioned = optim.EulerMaruyama(
        prox.L2(0.01),
        **settings,
        sampler_precond=linop.Identity((8, 8)),
        sampler_precond_diag=1.0,
        sampler_precond_tol=1e-4,
        sampler_precond_maxiter=5,
    )(y, A)
    assert torch.isfinite(preconditioned).all()
    assert not torch.equal(plain, preconditioned)


def test_the_sampler_reads_its_diagonal_first():
    # eulermaruyama_precond is entered on a positive diagonal and not on a
    # preconditioner, so one without the other would be silently ignored.
    with pytest.raises(ValueError, match="diagonal is positive"):
        optim.EulerMaruyama(prox.L2(0.01), step=0.1, sampler_precond=linop.Identity((8, 8)))


def test_a_zero_diagonal_leaves_the_plain_sampler():
    # A sampler draws from BART's own generator, so two runs never agree bit
    # for bit; what is checked is that a zero diagonal is the default and
    # takes the plain path rather than being refused or half-applied.
    A = linop.FFT((8, 8), axes=-1)
    y = A(_rand(8, 8))
    solver = optim.EulerMaruyama(prox.L2(0.01), step=0.1, maxiter=5)
    assert 0.0 == solver.sampler_precond_diag
    assert solver.sampler_precond is None
    made = optim.EulerMaruyama(prox.L2(0.01), step=0.1, maxiter=5, sampler_precond_diag=0.0)(y, A)
    assert torch.isfinite(made).all()


# --- what pics calls preconditioning ----------------------------------------


def test_the_tool_flag_is_a_regularizer_and_not_a_preconditioner():
    # `pics --precond` adds a weighted least-squares term with the inverse
    # sampling pattern as weights, chained through the model operator, and
    # never touches the linear solver.  It is a different thing with the same
    # name, and the tool is where it lives.
    # It is also only accepted by the two splitting algorithms, which is the
    # tell: a preconditioner would not care which iteration it was given to.
    n, coils = 12, 2
    maps = torch.ones(coils, 1, n, n, dtype=torch.complex64)
    kspace = _rand(coils, 1, n, n)
    pattern = torch.ones(1, 1, n, n, dtype=torch.complex64)
    regularized = dict(regularizers=prox.Wavelet((-1, -2), 0.001), i=5, m=True)
    plain = bt.pics(kspace, maps, **regularized)
    flagged = bt.pics(kspace, maps, **regularized, p=pattern, precond=True)
    assert torch.isfinite(flagged).all()
    assert not torch.equal(plain, flagged)
