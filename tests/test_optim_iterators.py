"""BART's proximal iterations, written as ``deepinv`` optimizers.

The claim is not that these converge to the same place.  It is that they are
the same iteration: every operator is BART's, the arithmetic is in the same
order, and what comes out is the library's answer to the bit, at every
iteration count rather than at one.

That is what makes them worth having in this form.  An iteration running
inside the library cannot be unrolled into a network or driven to a fixed
point by ``deepinv``, because there is nothing to differentiate through; one
written out here can be, and costs a few axpys on an image per step to say so.
"""

import pytest
import torch

import bartorch
from bartorch import linop, optim, prox

pytest.importorskip("deepinv")

from bartorch.optim import iterators  # noqa: E402


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


SHAPE = (1, 8, 8)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    return A, A(_rand(*SHAPE))


def _drive(iteration, A, y, term, iters, **params):
    """The loop ``BaseOptim`` runs, written out so the comparison is exact."""
    physics = bartorch.to_deepinv(A)
    fidelity = iterators.NormalEquations()
    prior = iterators.TermPrior(term, A.ishape)
    params = {"maxiter": iters, **params}

    X = {"est": (torch.zeros(*A.ishape, dtype=torch.complex64),) * 2}
    for _ in range(iters):
        X = iteration.forward(X, fidelity, prior, params, y, physics)
    return iteration.finish(X["est"][0], prior, params, iters)


# --- the same iteration, to the bit -------------------------------------------


@pytest.mark.parametrize("iters", [1, 2, 6, 13, 25, 60])
def test_ist_is_barts_ist(problem, iters):
    A, y = problem
    term = prox.L1(0.05)
    ours = _drive(iterators.ISTIteration(), A, y, term, iters, stepsize=0.7)
    assert torch.equal(ours, optim.IST(term, maxiter=iters, step=0.7)(y, A))


@pytest.mark.parametrize("iters", [1, 2, 6, 13, 25, 60])
def test_fista_is_barts_fista(problem, iters):
    """Thirteen is in the list because that is where it first went wrong.

    The ravine's coefficients are three single-precision operations in C, and
    working the same expression out in a double and rounding once is a
    different number; it takes thirteen iterations to reach the answer.
    """
    A, y = problem
    term = prox.L1(0.05)
    ours = _drive(iterators.FISTAIteration(), A, y, term, iters, stepsize=0.7)
    assert torch.equal(ours, optim.FISTA(term, maxiter=iters, step=0.7)(y, A))


@pytest.mark.parametrize("step", [0.25, 0.95, 1.5])
def test_the_step_is_the_one_bart_takes(problem, step):
    A, y = problem
    term = prox.Wavelet((-1, -2), 0.02)
    ours = _drive(iterators.FISTAIteration(), A, y, term, 20, stepsize=step)
    assert torch.equal(ours, optim.FISTA(term, maxiter=20, step=step)(y, A))


def test_the_acceleration_parameters_are_barts(problem):
    A, y = problem
    term = prox.L1(0.05)
    pqr = (1.0, 1.0, 2.0)
    ours = _drive(iterators.FISTAIteration(), A, y, term, 20, stepsize=0.7, pqr=pqr)
    assert torch.equal(ours, optim.FISTA(term, maxiter=20, step=0.7, pqr=pqr)(y, A))


@pytest.mark.parametrize("iters", [9, 11, 31, 40])
def test_hogwild_halves_the_step_where_bart_halves_it(problem, iters):
    """After ten steps, then twenty more, then forty.

    Held against FISTA rather than IST: `iter2_ist` asserts hogwild off, and
    an assertion in the library takes the process with it.
    """
    A, y = problem
    term = prox.L1(0.05)
    ours = _drive(iterators.FISTAIteration(), A, y, term, iters, stepsize=0.7, hogwild=True)
    assert torch.equal(ours, optim.FISTA(term, maxiter=iters, step=0.7, hogwild=True)(y, A))


def test_barts_iterative_soft_thresholding_refuses_hogwild():
    """It asserts rather than declines, and an assertion is the process."""
    with pytest.raises(ValueError, match="refuses hogwild"):
        optim.IST(prox.L1(0.05), hogwild=True)


@pytest.mark.parametrize("term", [prox.L1(0.05), prox.Wavelet((-1, -2), 0.02), prox.NonNegative()])
def test_any_term_that_thresholds_an_image_goes_through(problem, term):
    A, y = problem
    ours = _drive(iterators.FISTAIteration(), A, y, term, 15, stepsize=0.7)
    assert torch.equal(ours, optim.FISTA(term, maxiter=15, step=0.7)(y, A))


# --- the encoding keeps its own normal ----------------------------------------


def test_the_gradient_goes_through_the_encodings_own_normal():
    """`A^H A x - A^H y` rather than `A^H (A x - y)`.

    For a Toeplitz encoding the first is a convolution with a point spread
    function and the second is a transform and its adjoint -- a different
    route, and a measurably different answer.  Taking the obvious one would
    give up what the encoding was built for.
    """
    import bartorch.tools as bt

    n, coils = 16, 4
    torch.manual_seed(0)
    maps = _rand(coils, 1, n, n)
    traj = bt.traj(x=n, y=24, r=True)
    A = linop.NoncartesianSense(maps, (coils, n, n), traj=traj, toeplitz=True)
    y = _rand(*A.oshape)
    physics = bartorch.to_deepinv(A)

    x = _rand(*A.ishape)
    fidelity = iterators.NormalEquations()
    got = fidelity.grad(x, y, physics)

    # Against the cached adjoint rather than a fresh one: BART's gridding
    # reduces in whatever order its threads finish, so two adjoints of the
    # same data differ in the last bits.  Computing it once per solve is what
    # makes the iteration repeatable, not only what makes it quick.
    want = A.normal(x) - fidelity._adjoint_data(y, physics)
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    # And the route really is the convolution rather than the two transforms.
    assert (A.normal(x) - A.adjoint(A(x))).abs().max() / A.normal(x).abs().max() > 1e-5


def test_the_adjoint_data_is_computed_once_per_solve():
    """It does not change during one, and an extra adjoint a step is a
    transform a step."""
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    physics = bartorch.to_deepinv(A)

    calls = []
    plain = A.adjoint

    def counting(v, out=None):
        calls.append(1)
        return plain(v, out)

    A.adjoint = counting
    try:
        fidelity = iterators.NormalEquations()
        for _ in range(5):
            fidelity.grad(_rand(*SHAPE), y, physics)
    finally:
        del A.adjoint
    assert len(calls) == 1


# --- and they are deepinv optimizers ------------------------------------------


def test_an_iteration_drives_deepinvs_own_loop(problem):
    """Which is the point of the shape: `optim_builder` takes it, and so does
    everything built on `BaseOptim`."""
    from deepinv.optim import optim_builder

    A, y = problem
    solver = optim_builder(
        iteration=iterators.FISTAIteration(),
        data_fidelity=iterators.NormalEquations(),
        prior=iterators.TermPrior(prox.L1(0.05), A.ishape),
        params_algo={"stepsize": 0.7, "maxiter": 20, "lambda": 1.0, "g_param": 0.0},
        max_iter=20,
    )
    out = solver(y[None], bartorch.to_deepinv(A))
    assert out.shape == (1, *A.ishape)
    assert torch.isfinite(out).all()


def test_a_batch_is_walked_rather_than_folded_in(problem):
    """BART builds a term's operator per shape, and a batch is not one of its
    axes."""
    A, y = problem
    prior = iterators.TermPrior(prox.L1(0.05), A.ishape)
    batch = torch.stack([_rand(*A.ishape), _rand(*A.ishape)])
    out = prior.prox(batch, gamma=0.5)
    assert out.shape == batch.shape
    for item, got in zip(batch, out):
        torch.testing.assert_close(got, prox.L1(0.05).prox(item, 0.5), rtol=0, atol=0)


def test_a_deepinv_denoiser_can_stand_in_for_the_term(problem):
    """Plug and play: the iteration does not care where the prox comes from.

    An image here is complex and most of deepinv's denoisers are not, which is
    what `to_complex_denoiser` is for -- the denoiser runs on the real and
    imaginary parts separately.  Worth saying out loud, because a denoiser
    handed over without it fails inside itself rather than at the boundary.
    """
    from deepinv.models import MedianFilter, to_complex_denoiser
    from deepinv.optim import PnP

    A, y = problem
    physics = bartorch.to_deepinv(A)
    prior = PnP(denoiser=to_complex_denoiser(MedianFilter(kernel_size=3)))

    fidelity = iterators.NormalEquations()
    iteration = iterators.FISTAIteration()
    params = {"stepsize": 0.7, "maxiter": 5, "g_param": 0.05}

    x = torch.zeros(1, *A.ishape, dtype=torch.complex64)
    X = {"est": (x, x)}
    for _ in range(5):
        X = iteration.forward(X, fidelity, prior, params, y[None], physics)
    assert torch.isfinite(X["est"][0]).all()
