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
"""

import pytest
import torch

import bartorch.tools as bt
from bartorch import linop, optim, priors


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _ill_conditioned(n: int = 32):
    """A diagonal encoding whose singular values span two and a half decades."""
    scale = torch.logspace(0, -2.5, n, dtype=torch.float32).to(torch.complex64)
    return linop.Diagonal(scale, (n,)), scale


def _mixed(n: int = 16):
    """An encoding whose normal does not commute with a diagonal preconditioner,
    so the order the two are applied in shows in the answer."""
    torch.manual_seed(0)
    scale = torch.logspace(0, -1.5, n, dtype=torch.float32).to(torch.complex64)
    A = linop.Diagonal(scale, (n,)) @ linop.FFT((n,), axes=-1)
    M = linop.Diagonal(torch.linspace(0.5, 2.0, n).to(torch.complex64), (n,))
    return A, M, A(_rand(n))


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
        lambda M: optim.IST(priors.L1(0.001), maxiter=6, precond=M),
        lambda M: optim.FISTA(priors.L1(0.001), maxiter=6, precond=M),
        lambda M: optim.FISTA(priors.L1(0.001), maxiter=6, cclambda=0.1, precond=M),
        lambda M: optim.ADMM(priors.L1(0.001), maxiter=12, cg_maxiter=4, precond=M),
        lambda M: optim.ADMM(
            [priors.L1(0.001), priors.TotalVariation(-1, 0.001)],
            maxiter=12,
            cg_maxiter=4,
            dynamic_rho=True,
            precond=M,
        ),
        lambda M: optim.PRIDU(priors.L1(0.001), maxiter=6, precond=M),
        lambda M: optim.PRIDU(
            priors.TotalVariation(-1, 0.001), maxiter=6, adaptive_step=True, precond=M
        ),
    ],
    ids=["ist", "fista", "fista weight", "admm", "admm two terms", "pridu", "pridu adaptive"],
)
def test_every_loop_is_the_library_with_a_preconditioner(make):
    # lsqr2_create chains M onto the normal and the adjoint for every one of
    # these; the block does the same, in the same order.
    A, M, y = _mixed()
    solver = make(M)
    assert torch.equal(solver(y, A), solver._in_library(y, A))
    assert not torch.equal(solver(y, A), make(None)(y, A)), "the preconditioner changed nothing"


def test_conjugate_gradients_takes_one_too():
    A, M, y = _mixed()
    assert torch.isfinite(optim.CG(maxiter=6, precond=M)(y, A)).all()


def test_the_estimate_is_over_the_preconditioned_normal():
    from bartorch.optim.linear import maxeigen

    shape = (1, 8, 8)
    A = linop.FFT(shape, axes=(-1, -2))
    M = linop.Diagonal(torch.full(shape, 2.0, dtype=torch.complex64), shape)
    assert maxeigen(A) == pytest.approx(1.0, rel=1e-4)
    assert maxeigen(A, precond=M) == pytest.approx(2.0, rel=1e-4)


def test_a_step_from_the_estimate_is_the_librarys_with_a_preconditioner():
    # The power iteration starts from a random vector, so the two paths are
    # held close rather than to the bit.
    A, M, y = _mixed()
    solver = optim.FISTA(priors.L1(0.001), maxiter=8, eigen=True, precond=M)
    torch.testing.assert_close(solver(y, A), solver._in_library(y, A), rtol=1e-4, atol=1e-6)


def test_the_backward_solve_has_the_transposed_operator():
    A, M, _ = _mixed()
    block = optim.ADMMBlock(priors.TotalVariation(-1, 0.001), cclambda=0.1, precond=M)
    u, v = _rand(16), _rand(16)
    left = complex((v.conj() * block._xupdate_normal(A, 0.5, u)).sum())
    right = complex((block._xupdate_normal(A, 0.5, v, transposed=True).conj() * u).sum())
    assert abs(left - right) <= 1e-5 * abs(left)
    assert not torch.allclose(
        block._xupdate_normal(A, 0.5, u), block._xupdate_normal(A, 0.5, u, transposed=True)
    )


def test_a_preconditioned_solve_differentiates_to_its_data():
    A, M, y = _mixed()
    data = y.clone().requires_grad_()
    # A frozen term's last threshold is a constant, so a denoiser stands in the slot.
    made = optim.FISTA(priors.ImplicitPrior(lambda x: 0.9 * x), maxiter=4, precond=M)(data, A)
    made.abs().square().sum().backward()
    assert torch.isfinite(data.grad).all() and 0 < data.grad.abs().sum()


def test_a_python_defined_preconditioner_is_called_back():
    A, scale = _ill_conditioned(16)
    y = A(_rand(16))
    weight = (1.0 / scale.abs() ** 2).to(torch.complex64)
    seen = []

    def forward(x):
        seen.append(1)
        return weight * x

    M = linop.Callback((16,), (16,), forward, lambda v: weight.conj() * v)
    for solver in (optim.CG(maxiter=4, precond=M), optim.FISTA(priors.L1(0.001), precond=M)):
        seen.clear()
        made = solver(y, A)
        assert seen, f"{solver!r} never applied the preconditioner"
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
    regularized = dict(regularizers=priors.Wavelet((-1, -2), 0.001), i=5, m=True)
    plain = bt.pics(kspace, maps, **regularized)
    flagged = bt.pics(kspace, maps, **regularized, p=pattern, precond=True)
    assert torch.isfinite(flagged).all()
    assert not torch.equal(plain, flagged)
