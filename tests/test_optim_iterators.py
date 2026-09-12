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
    assert torch.equal(ours, optim.IST(term, maxiter=iters, step=0.7).in_library(y, A))


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
    assert torch.equal(ours, optim.FISTA(term, maxiter=iters, step=0.7).in_library(y, A))


@pytest.mark.parametrize("step", [0.25, 0.95, 1.5])
def test_the_step_is_the_one_bart_takes(problem, step):
    A, y = problem
    term = prox.Wavelet((-1, -2), 0.02)
    ours = _drive(iterators.FISTAIteration(), A, y, term, 20, stepsize=step)
    assert torch.equal(ours, optim.FISTA(term, maxiter=20, step=step).in_library(y, A))


def test_the_acceleration_parameters_are_barts(problem):
    A, y = problem
    term = prox.L1(0.05)
    pqr = (1.0, 1.0, 2.0)
    ours = _drive(iterators.FISTAIteration(), A, y, term, 20, stepsize=0.7, pqr=pqr)
    assert torch.equal(ours, optim.FISTA(term, maxiter=20, step=0.7, pqr=pqr).in_library(y, A))


@pytest.mark.parametrize("iters", [9, 11, 31, 40])
def test_hogwild_halves_the_step_where_bart_halves_it(problem, iters):
    """After ten steps, then twenty more, then forty.

    Held against FISTA rather than IST: `iter2_ist` asserts hogwild off, and
    an assertion in the library takes the process with it.
    """
    A, y = problem
    term = prox.L1(0.05)
    ours = _drive(iterators.FISTAIteration(), A, y, term, iters, stepsize=0.7, hogwild=True)
    assert torch.equal(
        ours, optim.FISTA(term, maxiter=iters, step=0.7, hogwild=True).in_library(y, A)
    )


def test_barts_iterative_soft_thresholding_refuses_hogwild():
    """It asserts rather than declines, and an assertion is the process."""
    with pytest.raises(ValueError, match="refuses hogwild"):
        optim.IST(prox.L1(0.05), hogwild=True)


@pytest.mark.parametrize("term", [prox.L1(0.05), prox.Wavelet((-1, -2), 0.02), prox.NonNegative()])
def test_any_term_that_thresholds_an_image_goes_through(problem, term):
    A, y = problem
    ours = _drive(iterators.FISTAIteration(), A, y, term, 15, stepsize=0.7)
    assert torch.equal(ours, optim.FISTA(term, maxiter=15, step=0.7).in_library(y, A))


# --- the public proximal solvers run these loops -------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda term, **kw: optim.IST(term, maxiter=12, step=0.7, **kw),
        lambda term, **kw: optim.FISTA(term, maxiter=12, step=0.7, **kw),
        lambda term, **kw: optim.FISTA(term, maxiter=12, step=0.7, hogwild=True, **kw),
        lambda term, **kw: optim.FISTA(term, maxiter=12, step=0.7, pqr=(1.0, 1.0, 2.0), **kw),
        lambda term, **kw: optim.PRIDU(term, maxiter=12, step=0.95, **kw),
        lambda term, **kw: optim.PRIDU(term, maxiter=12, step=0.95, adaptive_step=True, **kw),
        lambda term, **kw: optim.PRIDU(term, maxiter=12, step=0.95, sigma_tau_ratio=3.0, **kw),
    ],
    ids=["ist", "fista", "fista hogwild", "fista pqr", "pridu", "pridu adaptive", "pridu ratio"],
)
@pytest.mark.parametrize(
    "term",
    [prox.L1(0.05), prox.Wavelet((-1, -2), 0.02), prox.Laplace((-1, -2), 0.02)],
    ids=["l1", "wavelet", "laplace"],
)
@pytest.mark.parametrize("weight", [0.0, 0.1], ids=["no weight", "a quadratic weight"])
def test_the_public_solvers_are_the_library_they_replaced(make, term, weight):
    """`__call__` runs the iteration here; `in_library` runs BART's."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    solver = make(term, cclambda=weight)

    assert torch.equal(solver(y, A), solver.in_library(y, A))


def test_the_proximal_iterations_take_one_term():
    """`iter2_ist` and `iter2_fista` assert it, and an assertion is the
    process; this is the same refusal, said in Python."""
    with pytest.raises(ValueError, match="exactly one term"):
        optim.FISTA([prox.L1(0.05), prox.L1(0.02)], maxiter=5)
    with pytest.raises(ValueError, match="exactly one term"):
        optim.IST(maxiter=5)


def test_a_term_kept_across_solves_is_rewound_as_the_tool_rewinds_it():
    """A wavelet threshold's cycle spinning comes from a generator of its own,
    seeded when BART makes the operator.  The tool builds a fresh operator per
    run; a term here is kept, so every solve puts the generator back."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = prox.Wavelet((-1, -2), 0.02)

    for solver in (
        optim.FISTA(term, maxiter=8, step=0.7),
        optim.IST(term, maxiter=8, step=0.7),
        optim.ADMM(term, maxiter=8, cg_maxiter=4),
        optim.PRIDU(term, maxiter=8, step=0.95),
    ):
        assert torch.equal(solver(y, A), solver(y, A)), f"{solver!r} did not rewind its term"


def test_data_whose_adjoint_has_no_norm_is_left_alone():
    """`checkeps`: BART warns and returns without iterating."""
    A = linop.FFT(SHAPE, axes=(-1, -2))
    zero = torch.zeros(SHAPE, dtype=torch.complex64)
    solver = optim.FISTA(prox.L1(0.05), maxiter=8, step=0.7)
    assert torch.equal(solver(zero, A), solver.in_library(zero, A))


# --- the primal-dual split -----------------------------------------------------


def test_a_term_whose_transform_is_the_identity_becomes_the_primal_step():
    """Not a question about shapes: the Laplace term's transform has the
    image's shape and is a convolution, which is why BART asks the operator."""
    assert prox.L1(0.05).transform_is_identity(SHAPE)
    assert prox.Wavelet((-1, -2), 0.02).transform_is_identity(SHAPE)
    assert not prox.Laplace((-1, -2), 0.02).transform_is_identity(SHAPE)
    assert not prox.TotalVariation((-1, -2), 0.02).transform_is_identity(SHAPE)

    solver = optim.PRIDU([prox.L1(0.05), prox.TotalVariation((-1, -2), 0.01)], maxiter=6)
    primal, duals = solver._split(SHAPE)
    assert primal is solver.regularizers[0]
    assert duals == solver.regularizers[1:]

    solver = optim.PRIDU([prox.TotalVariation((-1, -2), 0.01), prox.L1(0.05)], maxiter=6)
    primal, duals = solver._split(SHAPE)
    assert primal is None
    assert duals == solver.regularizers


@pytest.mark.parametrize("steps", [1, 4, 12])
def test_several_terms_split_the_way_the_library_splits_them(steps):
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    terms = [prox.L1(0.05), prox.TotalVariation((-1, -2), 0.01)]
    solver = optim.PRIDU(terms, maxiter=steps, step=0.95)

    assert torch.equal(solver(y, A), solver.in_library(y, A))


@pytest.mark.parametrize(
    "term",
    [prox.TotalVariation((-1, -2), 0.01), prox.Laplace((-1, -2), 0.02)],
    ids=["total variation", "laplace"],
)
@pytest.mark.parametrize("steps", [1, 2, 6, 15])
def test_the_adaptive_step_with_a_dual_term_is_barts(term, steps):
    """The case that found the scalings.

    `chambolle_pock` works its coefficients out once, in a double, and rounds
    each to the float its vector is scaled by -- `1 / sigma`, `1 / (1 + sigma)`,
    `-sigma / (1 + sigma)`.  Dividing a tensor by `sigma` is a different
    number.  With `sigma` left where `pics` starts it the two agree; the
    adaptive step moves it by an order of magnitude, and then they do not.
    """
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    solver = optim.PRIDU(term, maxiter=steps, step=0.95, adaptive_step=True)

    assert torch.equal(solver(y, A), solver.in_library(y, A))


# --- the largest eigenvalue ----------------------------------------------------


def test_the_largest_eigenvalue_is_the_operator_the_step_divides_by():
    """`pics -e`.  The operator is the encoding's normal with the quadratic
    weight on its diagonal, and -- for the primal-dual iteration alone -- the
    dual terms' transforms added to it."""
    from bartorch.optim.linear import maxeigen

    n = 8
    assert maxeigen(linop.FFT((1, n, n), axes=(-1, -2))) == pytest.approx(1.0, rel=1e-5)
    assert maxeigen(linop.FFT((1, n, n), axes=(-1, -2)), cclambda=0.5) == pytest.approx(
        1.5, rel=1e-5
    )

    diag = torch.full((1, n, n), 0.1, dtype=torch.complex64)
    diag[0, 0, 0] = 2.0
    A = linop.Diagonal(diag, (1, n, n))
    assert maxeigen(A) == pytest.approx(4.0, rel=1e-5)

    # A gradient adds its own, which is what the primal-dual iteration
    # estimates over and the proximal ones do not.
    assert maxeigen(A, prox.TotalVariation((-1, -2), 0.01)) > 8.0


def test_the_estimate_is_a_random_draw_and_the_library_does_not_repeat_it():
    """Which is why the two paths are held close rather than to the bit when
    `eigen` is on: `estimate_maxeigenval` starts from a random vector, and
    BART's own answer changes from one solve to the next."""
    from bartorch.optim.linear import maxeigen

    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))

    draws = [maxeigen(A) for _ in range(4)]
    assert len(set(draws)) > 1

    solver = optim.FISTA(prox.L1(0.05), maxiter=12, step=0.7, eigen=True)
    assert not torch.equal(solver.in_library(y, A), solver.in_library(y, A))


@pytest.mark.parametrize(
    "make",
    [
        lambda term: optim.IST(term, maxiter=12, step=0.7, eigen=True),
        lambda term: optim.FISTA(term, maxiter=12, step=0.7, eigen=True),
        lambda term: optim.PRIDU(term, maxiter=12, step=0.95, eigen=True),
    ],
    ids=["ist", "fista", "pridu"],
)
def test_the_step_is_divided_by_the_estimate(make):
    torch.manual_seed(0)
    n = 8
    diag = torch.full((1, n, n), 0.1, dtype=torch.complex64)
    diag[0, 0, 0] = 2.0
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    term = prox.L1(0.05)

    ours = make(term)(y, A)
    torch.testing.assert_close(ours, make(term).in_library(y, A), rtol=1e-4, atol=1e-6)

    # And it is the estimate doing it: a step four times larger is a
    # different answer.
    assert not torch.equal(ours, optim.FISTA(term, maxiter=12, step=0.7)(y, A))


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


# --- alternating directions ---------------------------------------------------
#
# BART's ADMM solves `min 0.5||Ax-y||^2 + sum_j f_j(G_j x - b_j)`, which is the
# one iteration here that takes several terms, each with its own transform and
# its own bias.  Its x-update is conjugate gradients on
# `A^H A + rho sum_j G_j^H G_j`, warm-started, which is why it stays BART's
# even though the outer loop is not.


def _admm_steps(A, y, terms, steps, *, biases=None, rho=0.5, cg=10, **params):
    physics = bartorch.to_deepinv(A)
    fidelity = iterators.NormalEquations()
    iteration = iterators.ADMMIteration(terms, A.ishape, biases=biases)
    iteration.restart()
    settings = {"maxiter": 10**9, "cg_maxiter": cg, "rho": rho, **params}

    X = {"est": (torch.zeros(*A.ishape, dtype=torch.complex64),) * 2}
    for _ in range(steps):
        X = iteration.forward(X, fidelity, None, settings, y, physics)
    return X["est"][0]


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 8])
def test_admm_is_barts_admm(steps):
    """On a problem whose inner solve converges in one iteration, BART's
    budget and the outer step count are the same number, so the two can be
    held against each other directly."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = prox.L1(0.05)

    ours = _admm_steps(A, y, [term], steps)
    theirs = optim.ADMM(term, maxiter=steps, cg_maxiter=10, rho=0.5).in_library(y, A)
    assert torch.equal(ours, theirs)


def test_one_admm_step_is_barts_step_on_a_harder_problem():
    """Where the inner solve takes several iterations, BART's `maxiter` stops
    counting outer steps -- but the step itself is still the same one."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    term = prox.L1(0.05)

    ours = _admm_steps(A, y, [term], 1)
    theirs = optim.ADMM(term, maxiter=1, cg_maxiter=10, rho=0.5).in_library(y, A)
    assert torch.equal(ours, theirs)


def test_barts_budget_is_inner_applications_and_not_outer_steps():
    """Worth writing down, because `maxiter=30` does not mean thirty steps.

    `admm` breaks when `nr_invokes > maxiter`, and `nr_invokes` counts
    conjugate-gradient iterations across the whole run.
    """
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    base = linop.Diagonal(diag, (1, n, n))
    y = base(_rand(1, n, n))

    applications = []
    P = linop.Callback(
        (1, n, n),
        (1, n, n),
        base.forward,
        base.adjoint,
        lambda v: (applications.append(1), base.normal(v))[1],
    )
    optim.ADMM(prox.L1(0.05), maxiter=30, cg_maxiter=10, rho=0.5).in_library(y, P)

    # Thirty outer steps at ten inner iterations would be hundreds.
    assert 30 < len(applications) < 100


def test_total_variation_goes_through_a_gradient_an_operator_cannot_hold():
    """The term ADMM is for, and the one whose transform is rank seventeen.

    `transform()` refuses it; `apply_transform` is what the iteration uses,
    and the dot test says the pair it applies really are adjoint.
    """
    torch.manual_seed(0)
    term = prox.TotalVariation((-1, -2), 0.05)
    x = _rand(*SHAPE)
    gx = term.apply_transform(x, SHAPE)
    v = _rand(*gx.shape)

    assert gx.ndim == 17
    left = complex((gx.conj() * v).sum())
    right = complex((x.conj() * term.apply_transform(v, SHAPE, mode="adjoint")).sum())
    assert abs(left - right) / abs(left) < 1e-5

    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    out = _admm_steps(A, y, [term], 4)
    assert out.shape == A.ishape
    assert torch.isfinite(out.abs()).all()


def test_several_terms_are_split_apart():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    terms = [prox.L1(0.05), prox.Wavelet((-1, -2), 0.02)]
    out = _admm_steps(A, y, terms, 4)
    assert out.shape == A.ishape
    assert torch.isfinite(out.abs()).all()


# --- the public solver runs this loop ------------------------------------------
#
# `optim.ADMM` no longer hands the whole solve to `lsqr2`: it drives
# `ADMMIteration` step by step, which is what lets the same solver be unrolled.
# What follows is the claim that buys: the two paths answer with the same bits.


@pytest.mark.parametrize("budget", [1, 3, 7, 12, 30])
def test_the_public_solver_is_the_library_it_replaced(budget):
    """`ADMM.__call__` runs the iteration here; `ADMM.in_library` runs BART's."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    solver = optim.ADMM(prox.L1(0.05), maxiter=budget, cg_maxiter=4, rho=0.5)

    assert torch.equal(solver(y, A), solver.in_library(y, A))


def test_the_terms_are_summed_before_the_encoding_is_added():
    """`admm_normaleq` sums the terms first, scaling each by `rho` as it goes,
    and adds the encoding's normal last.

    With one term the two orders are the same two numbers added up, so they
    agree; with two they do not, and the difference is in the last place of
    the answer.  It is the sort of thing only a bit-for-bit comparison finds,
    so here is one with two terms in it.
    """
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    terms = [prox.L1(0.05), prox.L1(0.02)]
    solver = optim.ADMM(terms, maxiter=8, cg_maxiter=4, rho=0.5)

    assert torch.equal(solver(y, A), solver.in_library(y, A))


def test_without_deepinv_the_solver_falls_back_to_the_library(monkeypatch):
    """The iteration is a ``deepinv`` optimizer, and ``deepinv`` is optional.

    Without it there is nothing to drive, so ``ADMM`` runs BART's own loop --
    which is the same answer, as everything above this says.
    """

    def missing():
        raise ImportError("deepinv is not installed")

    monkeypatch.setattr(iterators, "_CACHE", {})
    monkeypatch.setattr(iterators, "_classes", missing)

    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    solver = optim.ADMM(prox.L1(0.05), maxiter=6, cg_maxiter=4)

    assert torch.equal(solver(y, A), solver.in_library(y, A))


def test_the_loop_crosses_into_python_where_the_library_would_not():
    """The point of the switch, and the cost of it: the step is visible."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))

    seen = []
    term = prox.L1(0.05)
    prox_of = term.prox
    term.prox = lambda *a, **kw: (seen.append(1), prox_of(*a, **kw))[1]

    optim.ADMM(term, maxiter=5, cg_maxiter=4)(y, A)
    assert seen, "the threshold was applied without passing through Python"


def test_a_bias_pulls_the_split_variable_towards_it():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = prox.L1(0.05)

    without = _admm_steps(A, y, [term], 4)
    with_bias = _admm_steps(A, y, [term], 4, biases=[_rand(*SHAPE)])
    assert (without - with_bias).abs().max() > 1e-3


def test_the_x_update_asks_the_encoding_for_its_own_normal():
    """A Toeplitz encoding stays one inside every ADMM step."""
    import bartorch.tools as bt

    n, coils = 16, 4
    torch.manual_seed(0)
    maps = _rand(coils, 1, n, n)
    traj = bt.traj(x=n, y=24, r=True)
    A = linop.NoncartesianSense(maps, (coils, n, n), traj=traj, toeplitz=True)
    y = _rand(*A.oshape)

    calls = []
    plain = A.normal

    def counting(v, out=None):
        calls.append(1)
        return plain(v, out)

    A.normal = counting
    try:
        _admm_steps(A, y, [prox.L1(0.01)], 2, cg=3)
    finally:
        del A.normal
    assert calls, "the encoding's own normal was never asked for"


# --- primal and dual ----------------------------------------------------------
#
# `chambolle_pock`, which `pics --pridu` runs.  The data term is carried as its
# own dual rather than differentiated, and each regularization term gets one
# too -- except the first, if its transform is the identity, which becomes the
# primal proximal step instead.  That split is `iter2_chambolle_pock`'s and is
# reproduced here rather than chosen.

import math  # noqa: E402


def _pridu_steps(A, y, steps, *, terms=(), primal=None, step=0.95, ratio=1.0, **extra):
    physics = bartorch.to_deepinv(A)
    fidelity = iterators.NormalEquations()
    iteration = iterators.PRIDUIteration(list(terms), A.ishape, primal=primal)
    params = {
        "sigma": math.sqrt(step) * ratio,
        "tau": math.sqrt(step) / ratio,
        "sigma_tau_ratio": ratio,
        "maxiter": steps,
        **extra,
    }
    X = {"est": (torch.zeros(*A.ishape, dtype=torch.complex64),) * 2}
    for _ in range(steps):
        X = iteration.forward(X, fidelity, None, params, y, physics)
        if X["done"]:
            break
    return X["est"][0]


@pytest.mark.parametrize("steps", [1, 2, 5, 10, 25, 60])
def test_pridu_is_barts_pridu(problem, steps):
    """The term's transform is the identity, so it is the primal step."""
    A, y = problem
    term = prox.L1(0.05)
    ours = _pridu_steps(A, y, steps, primal=term)
    assert torch.equal(ours, optim.PRIDU(term, maxiter=steps, step=0.95).in_library(y, A))


@pytest.mark.parametrize("steps", [1, 3, 8])
def test_a_term_with_a_transform_becomes_a_dual(problem, steps):
    """Total variation's is a gradient, so it cannot be the primal step and
    `prox2` falls back to the identity, as `prox_zero_create` is."""
    A, y = problem
    term = prox.TotalVariation((-1, -2), 0.05)
    ours = _pridu_steps(A, y, steps, terms=[term])
    assert torch.equal(ours, optim.PRIDU(term, maxiter=steps, step=0.95).in_library(y, A))


@pytest.mark.parametrize("steps", [1, 3, 8, 20])
def test_hogwild_is_a_decay_here_rather_than_a_halving(problem, steps):
    """0.95 a step, and `(float)pow(decay, i)` from a float32 `decay` -- taken
    in double and rounded once, which is not the same as taking it in one."""
    A, y = problem
    term = prox.L1(0.05)
    ours = _pridu_steps(A, y, steps, primal=term, decay=0.95)
    assert torch.equal(
        ours, optim.PRIDU(term, maxiter=steps, step=0.95, hogwild=True).in_library(y, A)
    )


@pytest.mark.parametrize("steps", [1, 3, 8])
def test_the_adaptive_step_is_barts(problem, steps):
    A, y = problem
    term = prox.L1(0.05)
    ours = _pridu_steps(A, y, steps, primal=term, adaptive_step=True)
    assert torch.equal(
        ours, optim.PRIDU(term, maxiter=steps, step=0.95, adaptive_step=True).in_library(y, A)
    )


@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_the_step_ratio_splits_sigma_and_tau_the_way_pics_does(problem, ratio):
    A, y = problem
    term = prox.L1(0.05)
    ours = _pridu_steps(A, y, 10, primal=term, ratio=ratio)
    assert torch.equal(
        ours, optim.PRIDU(term, maxiter=10, step=0.95, sigma_tau_ratio=ratio).in_library(y, A)
    )
