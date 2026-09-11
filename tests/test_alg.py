"""The iterations and the proximal steps, against what they are supposed to be.

Each algorithm is checked against something outside itself: a closed form, a
direct solve, or BART's own answer to the same problem.  An iteration that
agrees only with another iteration proves nothing.
"""

import pytest
import torch

import bartorch.tools as bt
from bartorch import alg, linop, prox


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _encoding(coils=4, n=16):
    """A Cartesian SENSE encoding with orthonormal maps, so A^H A is the identity."""
    maps = _rand(coils, n, n)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    return linop.MultiplySum(maps, (1, n, n), (coils, n, n)), maps


# --- the proximal operators -------------------------------------------------


def test_soft_thresholding_shrinks_the_magnitude_and_keeps_the_phase():
    x = _rand(64)
    out = prox.soft_threshold(x, 0.5)
    kept = x.abs() > 0.5
    torch.testing.assert_close(out.abs()[kept], x.abs()[kept] - 0.5, rtol=1e-4, atol=1e-5)
    assert (out.abs()[~kept] == 0).all()
    torch.testing.assert_close(
        torch.angle(out[kept]), torch.angle(x[kept]), rtol=1e-4, atol=1e-5
    )


def test_the_l1_prox_is_the_minimiser_it_says_it_is():
    """Checked against the objective directly, on a grid around the answer."""
    x = torch.tensor([1.0 + 0.5j], dtype=torch.complex64)
    alpha, weight = 0.4, 1.0
    got = prox.L1((1,), weight=weight)(alpha, x)

    def objective(z):
        return weight * z.abs().sum() + (z - x).abs().square().sum() / (2 * alpha)

    here = objective(got)
    for step in (-0.1, -0.01, 0.01, 0.1):
        for direction in (1.0, 1.0j):
            assert objective(got + step * direction) >= here - 1e-5


def test_the_zero_prox_is_the_identity():
    x = _rand(8)
    torch.testing.assert_close(prox.Zero((8,))(0.5, x), x)


def test_the_squared_l2_prox_shrinks_toward_its_centre():
    x = _rand(8)
    centre = _rand(8)
    got = prox.L2Squared((8,), weight=1.0, centre=centre)(0.5, x)
    # With alpha and weight both making the shrinkage one half.
    torch.testing.assert_close(got, centre + (x - centre) / 2, rtol=1e-4, atol=1e-5)


def test_the_ball_prox_projects_onto_the_ball():
    inside, outside = _rand(8) * 0.01, _rand(8) * 100
    ball = prox.L2Ball((8,), radius=1.0)
    torch.testing.assert_close(ball(1.0, inside), inside)
    assert ball(1.0, outside).flatten().norm().item() == pytest.approx(1.0, rel=1e-5)


def test_a_stack_applies_each_to_its_own_part():
    stacked = prox.Stack([prox.L1((4,), weight=10.0), prox.Zero((4,))])
    x = _rand(8)
    got = stacked(1.0, x)
    assert (got[:4].abs() == 0).all()  # thresholded away
    torch.testing.assert_close(got[4:], x[4:])


# --- the iterations ---------------------------------------------------------


def test_conjugate_gradients_solves_what_bart_solves():
    """Against BART's own, which is the same problem and a different loop."""
    A, _ = _encoding()
    truth = _rand(1, 16, 16)
    data = A(truth)
    ours = alg.ConjugateGradient(A, data, lambda_=0.0, max_iter=40).run()
    theirs = A.lstsq(data, maxiter=40)
    torch.testing.assert_close(ours, theirs, rtol=1e-3, atol=1e-4)


def test_conjugate_gradients_recovers_the_image_through_an_orthonormal_encoding():
    A, _ = _encoding()
    truth = _rand(1, 16, 16)
    got = alg.ConjugateGradient(A, A(truth), max_iter=40).run()
    torch.testing.assert_close(got, truth, rtol=1e-3, atol=1e-4)


def test_a_tikhonov_weight_shrinks_the_answer():
    """With A^H A the identity, the solution is the data over one plus lambda."""
    A, _ = _encoding()
    truth = _rand(1, 16, 16)
    got = alg.ConjugateGradient(A, A(truth), lambda_=1.0, max_iter=60).run()
    torch.testing.assert_close(got, truth / 2, rtol=1e-2, atol=1e-3)


def test_gradient_descent_gets_there_too_given_enough_steps():
    A, _ = _encoding()
    truth = _rand(1, 16, 16)
    got = alg.GradientDescent(A, A(truth), max_iter=300, tol=0).run()
    error = (got - truth).flatten().norm() / truth.flatten().norm()
    assert error < 0.05


def test_proximal_gradient_with_no_penalty_is_gradient_descent():
    A, _ = _encoding()
    truth = _rand(1, 16, 16)
    data = A(truth)
    plain = alg.GradientDescent(A, data, max_iter=50, tol=0).run()
    proximal = alg.ProximalGradient(
        A, data, prox.Zero(A.ishape), accelerate=False, max_iter=50, tol=0
    ).run()
    torch.testing.assert_close(plain, proximal, rtol=1e-4, atol=1e-5)


def test_acceleration_lowers_the_objective_faster():
    """Which is the whole claim FISTA makes over ISTA, and it is a claim about
    the objective rather than about a truth the method does not know.

    On an ill-conditioned operator, because that is the only place there is
    anything to improve: with orthonormal maps the normal operator is the
    identity, the step is one, and proximal gradient lands on the answer in a
    single step whether it is accelerated or not.
    """
    torch.manual_seed(0)
    n = 256
    diagonal = torch.logspace(0, -2.5, n).to(torch.complex64)
    A = linop.Diagonal(diagonal, (n,))
    data = A(_rand(n))
    weight = 1e-3
    g = prox.L1(A.ishape, weight=weight)

    def objective(x):
        return float((A(x) - data).abs().square().sum() / 2 + weight * x.abs().sum())

    for steps in (10, 20, 40):
        slower = alg.ProximalGradient(A, data, g, accelerate=False, max_iter=steps, tol=0)
        faster = alg.ProximalGradient(A, data, g, accelerate=True, max_iter=steps, tol=0)
        assert objective(faster.run()) < objective(slower.run()), steps


def test_an_l1_penalty_leaves_a_sparse_answer_sparser():
    A, _ = _encoding()
    truth = torch.zeros(1, 16, 16, dtype=torch.complex64)
    truth[0, 4, 4] = 5.0
    truth[0, 9, 11] = 3.0
    data = A(truth)
    got = alg.ProximalGradient(
        A, data, prox.L1(A.ishape, weight=0.2), max_iter=120, tol=0
    ).run()
    assert (got.abs() > 1e-3).sum() < 20
    assert got[0, 4, 4].abs() > 2.0


def test_the_step_is_estimated_when_it_is_not_given():
    """One over the largest eigenvalue of the normal operator; for orthonormal
    maps that is one."""
    A, _ = _encoding()
    assert alg.max_eigenvalue_step(A, torch.zeros(A.ishape)) == pytest.approx(1.0, rel=1e-3)


# --- what the torch loop is for ---------------------------------------------


def test_an_iteration_is_a_step_so_a_caller_can_drive_it():
    A, _ = _encoding()
    truth = _rand(1, 16, 16)
    solver = alg.ConjugateGradient(A, A(truth), max_iter=1000)
    for _ in range(5):
        solver.update()
    assert solver.iteration == 5
    assert not torch.equal(solver.x, torch.zeros_like(solver.x))


def test_a_gradient_comes_back_through_the_unrolled_loop():
    """The reason these are written in torch: a few steps of one is a layer,
    and the gradient reaches the data through the encoding."""
    A, _ = _encoding(coils=2, n=8)
    data = _rand(2, 8, 8).requires_grad_(True)
    out = alg.ProximalGradient(
        A, data, prox.L1(A.ishape, weight=1e-3), max_iter=4, tol=0
    ).run()
    out.abs().square().sum().backward()
    assert data.grad is not None
    assert data.grad.shape == data.shape
    assert torch.isfinite(data.grad).all()


def test_a_learned_weight_in_the_penalty_gets_a_gradient():
    """An unrolled network with a trainable regularisation strength."""
    A, _ = _encoding(coils=2, n=8)
    truth = _rand(1, 8, 8)
    data = A(truth)
    weight = torch.tensor(0.05, requires_grad=True)

    class Learned(prox.Prox):
        shape = A.ishape

        def __call__(self, alpha, x):
            return prox.soft_threshold(x, weight * alpha)

    out = alg.ProximalGradient(A, data, Learned(), max_iter=4, tol=0).run()
    (out - truth).abs().square().sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad)


def test_the_same_problem_through_barts_own_solver_agrees():
    """The torch loop and `pics` are two ways at one answer, and should meet."""
    kspace = bt.phantom(24, coils=4, kspace=True)
    maps = bt.ecalib(kspace, maps=1).squeeze(1)
    A = linop.MultiplySum(maps, (1, 24, 24), (4, 24, 24))
    coil_images = bt.ifft(kspace, axes=(-1, -2), unitary=True).squeeze(1)
    ours = alg.ConjugateGradient(A, coil_images, lambda_=0.01, max_iter=40).run()
    theirs = A.lstsq(coil_images, lambda_=0.01, maxiter=40)
    torch.testing.assert_close(ours, theirs, rtol=1e-2, atol=1e-3)
