"""The solve inside a network: what carries a gradient, and what that gradient is.

Every operator application in :mod:`bartorch` is already recorded for autograd,
so an iteration assembled out of applications, axpys and a proximal operator is
a torch graph over BART's arithmetic.  The one thing that was not was the
linear solve: conjugate gradients runs inside the library and hands back an
array with no history.

It is now, by implicit differentiation -- ``x = N^-1 A^H y`` is linear in ``y``,
so the backward pass is one more solve with the same operator.  That is what
BART does for the one solver it made an ``nlop``: ``norm_inv_der_src`` and
``norm_inv_adj_src`` in ``src/nlops/norm_inv.c`` each run a conjugate-gradient
solve of their own rather than unrolling the forward one.

The consequence worth stating is that the recorded gradient is the derivative
of the *solution*, not of the truncated iteration that approximated it.  With
the solve converged the two are the same thing, and
``test_the_gradient_is_the_solves_and_not_the_iterations_it_took`` is what says
so where they are not.
"""

import math
import warnings

import pytest
import torch

from bartorch import linop, optim, priors

SHAPE = (1, 8, 8)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _dense(A, shape):
    """``A`` as a matrix, by applying it to a basis."""
    n = math.prod(shape)
    columns = [A.forward(e.reshape(shape)).reshape(-1) for e in torch.eye(n, dtype=torch.complex64)]
    return torch.stack(columns, dim=1)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    return A, A(_rand(*SHAPE))


class _Scale:
    """A denoiser with one learnable number in it, which is all a gradient needs."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, sigma=None):
        return self.weight * x


# --- the solve itself ---------------------------------------------------------


def test_the_solve_differentiates_to_its_data():
    # x = (A^H A + q)^-1 A^H y, against the same expression as a dense solve.
    torch.manual_seed(0)
    shape, q = (6,), 0.5
    A = linop.FFT(shape, axes=-1)
    M = _dense(A, shape)
    N = M.conj().T @ M + q * torch.eye(6, dtype=torch.complex64)
    y = _rand(6)

    ours = y.clone().requires_grad_(True)
    optim.CG(maxiter=200, cclambda=q)(ours, A).abs().square().sum().backward()

    dense = y.clone().requires_grad_(True)
    torch.linalg.solve(N, M.conj().T @ dense).abs().square().sum().backward()

    torch.testing.assert_close(ours.grad, dense.grad, rtol=1e-4, atol=1e-5)


def test_the_tikhonov_weight_is_in_the_backward_solve_too():
    # `lambda_` reaches the normal operator as a term rather than as `cclambda`,
    # and a backward pass that dropped it would invert a different matrix.
    torch.manual_seed(0)
    shape, weight = (6,), 0.7
    A = linop.FFT(shape, axes=-1)
    M = _dense(A, shape)
    N = M.conj().T @ M + weight * torch.eye(6, dtype=torch.complex64)
    y = _rand(6)

    ours = y.clone().requires_grad_(True)
    optim.CG(weight, maxiter=200)(ours, A).abs().square().sum().backward()

    dense = y.clone().requires_grad_(True)
    torch.linalg.solve(N, M.conj().T @ dense).abs().square().sum().backward()

    torch.testing.assert_close(ours.grad, dense.grad, rtol=1e-4, atol=1e-5)


def test_the_forward_pass_is_the_same_bits_either_way(problem):
    """Recording the solve must not change it: the answer a training run sees
    and the answer a reconstruction sees are the same number."""
    A, y = problem
    plain = optim.CG(maxiter=12, cclambda=0.1)(y, A)
    recorded = optim.CG(maxiter=12, cclambda=0.1)(y.clone().requires_grad_(True), A)
    assert torch.equal(plain, recorded.detach())


def test_a_solve_asked_for_no_gradient_is_not_recorded(problem):
    A, y = problem
    with torch.no_grad():
        made = optim.CG(maxiter=8)(y.clone().requires_grad_(True), A)
    assert made.grad_fn is None


def test_the_gradient_is_the_solves_and_not_the_iterations_it_took():
    """Six iterations and two hundred give the same gradient, because what is
    differentiated is the linear system and not the walk towards it.

    That is the implicit-differentiation trade and it is worth having in a
    test: a truncated forward pass gets the converged derivative, which is
    what makes the backward pass cost a solve rather than a tape of one.
    """
    torch.manual_seed(0)
    shape = (6,)
    A = linop.Diagonal(torch.linspace(0.6, 1.4, 6).to(torch.complex64), shape)
    y = _rand(6)

    grads = []
    for iterations in (6, 200):
        data = y.clone().requires_grad_(True)
        optim.CG(maxiter=iterations, cclambda=0.4)(data, A).abs().square().sum().backward()
        grads.append(data.grad)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_a_warm_start_carries_no_gradient(problem):
    """``x0`` is where the iteration began, and the solution of a linear system
    does not depend on that -- so it is the constant it mathematically is."""
    A, y = problem
    start = torch.zeros(A.ishape, dtype=torch.complex64, requires_grad=True)
    optim.CG(maxiter=8, cclambda=0.1)(
        y.clone().requires_grad_(True), A, start
    ).abs().sum().backward()
    assert start.grad is None


def test_a_python_defined_operator_differentiates_through_the_solve():
    """The backward solve asks the operator for its normal, and an operator
    written here answers with two callbacks rather than a BART operator."""
    torch.manual_seed(0)
    shape = (6,)
    weights = torch.linspace(0.6, 1.4, 6).to(torch.complex64)
    A = linop.Callback(shape, shape, lambda x: weights * x, lambda v: weights.conj() * v)

    ours = _rand(6).requires_grad_(True)
    optim.CG(maxiter=120, cclambda=0.3)(ours, A).abs().square().sum().backward()

    N = weights.abs() ** 2 + 0.3
    dense = ours.detach().clone().requires_grad_(True)
    (weights.conj() * dense / N).abs().square().sum().backward()
    torch.testing.assert_close(ours.grad, dense.grad, rtol=1e-4, atol=1e-5)


def test_real_data_gets_a_real_gradient():
    torch.manual_seed(0)
    shape = (6,)
    A = linop.Diagonal(torch.linspace(0.6, 1.4, 6).to(torch.complex64), shape)
    y = torch.randn(6, requires_grad=True)
    optim.CG(maxiter=40, cclambda=0.3)(y, A).abs().square().sum().backward()
    assert y.grad.dtype == y.dtype
    assert not y.grad.is_complex()


# --- the iterations, unrolled -------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda y, A, t: optim.ist(y, A, t, maxiter=3, step=0.7),
        lambda y, A, t: optim.fista(y, A, t, maxiter=3, step=0.7),
        lambda y, A, t: optim.pridu(y, A, t, maxiter=3, step=0.95),
        lambda y, A, t: optim.admm(y, A, t, maxiter=2, cg_maxiter=8, rho=0.5),
    ],
    ids=["ist", "fista", "pridu", "admm"],
)
def test_every_iteration_reaches_a_parameter_in_the_denoiser_slot(problem, call):
    """Which is the whole point of the iterations living here: unroll one and
    the thing standing where a term goes is trained by whatever trains the
    network around it."""
    A, y = problem
    weight = torch.nn.Parameter(torch.tensor(0.8))
    made = call(y, A, priors.ImplicitPrior(_Scale(weight)))
    assert made.grad_fn is not None
    made.abs().square().sum().backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad) and 0.0 != weight.grad


@pytest.mark.parametrize(
    "call",
    [
        lambda y, A, t: optim.fista(y, A, t, maxiter=3, step=0.7),
        lambda y, A, t: optim.admm(y, A, t, maxiter=3, cg_maxiter=40, rho=0.5),
    ],
    ids=["fista", "admm"],
)
def test_the_unrolled_gradient_is_the_one_finite_differences_measure(problem, call):
    A, y = problem

    def loss(value):
        weight = torch.as_tensor(value, dtype=torch.float32)
        return call(y, A, priors.ImplicitPrior(_Scale(weight))).abs().square().sum()

    weight = torch.tensor(0.8, requires_grad=True)
    loss(weight).backward()

    h = 1e-3
    measured = (loss(0.8 + h) - loss(0.8 - h)).item() / (2 * h)
    assert abs(weight.grad.item() - measured) <= 1e-3 * abs(measured)


def test_the_data_carries_a_gradient_through_a_whole_unrolled_solve(problem):
    A, y = problem
    data = y.clone().requires_grad_(True)
    made = optim.admm(
        data, A, priors.ImplicitPrior(_Scale(torch.tensor(0.8))), maxiter=2, cg_maxiter=8
    )
    made.abs().square().sum().backward()
    assert data.grad is not None
    assert torch.isfinite(data.grad).all()


def test_a_first_admm_step_does_not_depend_on_the_prior_and_says_nothing_else():
    """Worth writing down.  ``admm`` solves for ``x`` before it has thresholded
    anything, so after one step the answer really is independent of whatever
    stands in the term's slot -- and torch says that by refusing to run a
    backward pass at all rather than by handing back a zero."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    weight = torch.nn.Parameter(torch.tensor(0.8))
    one = optim.admm(y, A, priors.ImplicitPrior(_Scale(weight)), maxiter=1, cg_maxiter=8)
    assert one.grad_fn is None
    two = optim.admm(y, A, priors.ImplicitPrior(_Scale(weight)), maxiter=2, cg_maxiter=8)
    assert two.grad_fn is not None


# --- what has no derivative to give -------------------------------------------


def test_a_bart_term_refuses_rather_than_giving_a_wrong_gradient(problem):
    A, y = problem
    data = y.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="carries no derivative"):
        optim.admm(data, A, priors.L1(0.01), maxiter=2, cg_maxiter=8)


def test_a_frozen_term_says_that_is_what_was_meant(problem):
    A, y = problem
    data = y.clone().requires_grad_(True)
    made = optim.admm(data, A, priors.frozen(priors.L1(0.01)), maxiter=2, cg_maxiter=8)
    made.abs().square().sum().backward()
    assert torch.isfinite(data.grad).all()


def test_freezing_a_term_changes_no_numbers(problem):
    """Detaching is not an approximation of the forward pass -- it is the same
    arithmetic -- so the library route is still there and still exact."""
    A, y = problem
    settings = dict(maxiter=3, cg_maxiter=8)
    plain = optim.ADMM(priors.L1(0.01), **settings)
    frozen = optim.ADMM(priors.frozen(priors.L1(0.01)), **settings)
    assert torch.equal(frozen(y, A), plain(y, A))
    assert torch.equal(frozen._in_library(y, A), plain._in_library(y, A))


def test_the_transform_in_front_of_a_term_is_recorded(problem):
    """Total variation thresholds the components of a gradient, so the step
    applies a finite difference to the iterate and its adjoint on the way
    back.  Both are BART's, and both are in the graph."""
    A, y = problem
    term = priors.frozen(priors.TotalVariation((-1, -2), 0.01))
    weight = torch.nn.Parameter(torch.tensor(0.8))
    # `maxiter` is a budget on conjugate-gradient iterations across the whole
    # run, not a count of steps, and a gradient needs more than one step.
    made = optim.admm(y, A, [priors.ImplicitPrior(_Scale(weight)), term], maxiter=24, cg_maxiter=4)
    made.abs().square().sum().backward()
    assert weight.grad is not None and 0.0 != weight.grad


def test_the_iteration_does_not_warn_about_its_own_residuals(problem):
    """The residual norms are read as Python floats to steer ``rho`` and the
    stopping test.  They are detached on purpose, and torch is not left to
    warn once a step that someone might have meant otherwise."""
    A, y = problem
    weight = torch.nn.Parameter(torch.tensor(0.8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optim.admm(y, A, priors.ImplicitPrior(_Scale(weight)), maxiter=2, cg_maxiter=8, rho=0.5)
        optim.pridu(y, A, priors.ImplicitPrior(_Scale(weight)), maxiter=3, step=0.95)
    assert [] == [w for w in caught if "requires_grad" in str(w.message)]
