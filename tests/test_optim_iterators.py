"""BART's proximal iterations, one block per step, held against the library.

The claim is not that these converge to the same place.  It is that they are
the same iteration: every operator is BART's, the arithmetic is in the same
order, and what comes out is the library's answer to the bit, at every
iteration count rather than at one.
"""

import functools

import numpy as np
import pytest
import torch

from bartorch import linop, optim, priors


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


SHAPE = (1, 8, 8)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    return A, A(_rand(*SHAPE))


def _drive(block, A, y, iters):
    """A block looped by hand, so what is held against the library is the step."""
    state = block.start(y, A)
    for _ in range(iters):
        state = block(state, A)
    return block.output(state, A)


# --- the same iteration, to the bit -------------------------------------------


@pytest.mark.parametrize("iters", [1, 2, 6, 13, 25, 60])
def test_ist_is_barts_ist(problem, iters):
    A, y = problem
    term = priors.L1(0.05)
    ours = _drive(optim.ISTBlock(term, step=0.7), A, y, iters)
    assert torch.equal(ours, optim.IST(term, maxiter=iters, step=0.7)._in_library(y, A))


@pytest.mark.parametrize("iters", [1, 2, 6, 13, 25, 60])
def test_fista_is_barts_fista(problem, iters):
    """Thirteen is in the list because that is where it first went wrong.

    The ravine's coefficients are three single-precision operations in C, and
    working the same expression out in a double and rounding once is a
    different number; it takes thirteen iterations to reach the answer.
    """
    A, y = problem
    term = priors.L1(0.05)
    ours = _drive(optim.FISTABlock(term, step=0.7), A, y, iters)
    assert torch.equal(ours, optim.FISTA(term, maxiter=iters, step=0.7)._in_library(y, A))


@pytest.mark.parametrize("step", [0.25, 0.95, 1.5])
def test_the_step_is_the_one_bart_takes(problem, step):
    A, y = problem
    term = priors.Wavelet((-1, -2), 0.02)
    ours = _drive(optim.FISTABlock(term, step=step), A, y, 20)
    assert torch.equal(ours, optim.FISTA(term, maxiter=20, step=step)._in_library(y, A))


def test_the_acceleration_parameters_are_barts(problem):
    """`fista_formula` is `(p + sqrtf(q + r t^2)) / 2`, and whether that is
    one rounding or two is the compiler's choice.

    clang contracts `q + r * t * t` into a fused multiply-add where the
    hardware has one -- arm64 does, the x86-64 baseline does not -- and a
    fused multiply-add does not round the product.  `_formula` writes the
    two roundings out, which is what BART computes on x86-64.

    With BART's own `r = 4` the two forms agree at every step of the
    recurrence, so the default is the same bits everywhere; `r = 2` parts
    company at the eighth.  Hence the two assertions: exact for the
    parameters `pics` uses, and close for parameters that reach the
    difference between one rounding and two.
    """
    A, y = problem
    term = priors.L1(0.05)

    theirs = optim.FISTA(term, maxiter=20, step=0.7)._in_library(y, A)
    ours = _drive(optim.FISTABlock(term, step=0.7, pqr=(1.0, 1.0, 4.0)), A, y, 20)
    assert torch.equal(ours, theirs), "BART's own acceleration parameters are not platform-bound"

    pqr = (1.0, 1.0, 2.0)
    ours = _drive(optim.FISTABlock(term, step=0.7, pqr=pqr), A, y, 20)
    theirs = optim.FISTA(term, maxiter=20, step=0.7, pqr=pqr)._in_library(y, A)
    assert not torch.equal(ours, optim.FISTA(term, maxiter=20, step=0.7)._in_library(y, A)), (
        "the parameters changed nothing, so this proves nothing about them"
    )
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


def test_the_acceleration_recurrence_is_where_the_platform_shows():
    """The measurement the test above rests on, without a solve around it.

    Not a claim about this package: it is what a fused multiply-add does to
    BART's recurrence, and it says which parameters can be held to the bit on
    every platform and which cannot.
    """
    import numpy as np

    def contracted(q, r, t):
        # float32 inputs make the double exact, so this is the fused form.
        t32 = np.float32(t)
        return np.float32(np.float64(q) + np.float64(np.float32(r) * t32) * np.float64(t32))

    def rounded(q, r, t):
        t32 = np.float32(t)
        return np.float32(np.float32(q) + np.float32(r) * t32 * t32)

    def step(inner):
        return float((np.float32(1.0) + np.sqrt(inner)) / np.float32(2.0))

    # Twenty, because that is how many the test above runs.
    counts = {}
    for r in (4.0, 2.0):
        t, differing = 1.0, 0
        for _ in range(20):
            one, two = step(rounded(1.0, r, t)), step(contracted(1.0, r, t))
            differing += int(one != two)
            t = one
        counts[r] = differing

    assert counts[4.0] == 0, (
        "BART's own r is 4, and the exact assertion above rests on the two "
        f"forms agreeing at every step of it -- they part at {counts[4.0]}"
    )
    assert counts[2.0] > 0, (
        "r = 2 was chosen because it reaches the difference between one "
        "rounding and two; it no longer does, so the tolerance above is idle"
    )


@pytest.mark.parametrize("iters", [9, 11, 31, 40])
def test_hogwild_halves_the_step_where_bart_halves_it(problem, iters):
    """After ten steps, then twenty more, then forty.

    Held against FISTA rather than IST: `iter2_ist` asserts hogwild off, and
    an assertion in the library takes the process with it.
    """
    A, y = problem
    term = priors.L1(0.05)
    ours = _drive(optim.FISTABlock(term, step=0.7, hogwild=True), A, y, iters)
    assert torch.equal(
        ours, optim.FISTA(term, maxiter=iters, step=0.7, hogwild=True)._in_library(y, A)
    )


def test_barts_iterative_soft_thresholding_refuses_hogwild():
    """It asserts rather than declines, and an assertion is the process."""
    with pytest.raises(ValueError, match="refuses hogwild"):
        optim.IST(priors.L1(0.05), hogwild=True)


@pytest.mark.parametrize(
    "term", [priors.L1(0.05), priors.Wavelet((-1, -2), 0.02), priors.NonNegative()]
)
def test_any_term_that_thresholds_an_image_goes_through(problem, term):
    A, y = problem
    ours = _drive(optim.FISTABlock(term, step=0.7), A, y, 15)
    assert torch.equal(ours, optim.FISTA(term, maxiter=15, step=0.7)._in_library(y, A))


# --- the public proximal solvers run these loops -------------------------------


def _solver_agrees(solver, ours, theirs):
    """To the bit, except where the library's own multiply-add was fused.

    `_pridu_agrees` says why that exception exists and which builds it
    applies to; every other iteration is the same bits everywhere.
    """
    if isinstance(solver, optim.PRIDU):
        _pridu_agrees(ours, theirs)
    else:
        assert torch.equal(ours, theirs)


_THRESHOLDING = {
    "ist": lambda term, **kw: optim.IST(term, maxiter=12, step=0.7, **kw),
    "fista": lambda term, **kw: optim.FISTA(term, maxiter=12, step=0.7, **kw),
    "fista hogwild": lambda term, **kw: optim.FISTA(term, maxiter=12, step=0.7, hogwild=True, **kw),
    "fista pqr": lambda term, **kw: optim.FISTA(
        term, maxiter=12, step=0.7, pqr=(1.0, 1.0, 2.0), **kw
    ),
}
_PRIMAL_DUAL = {
    "pridu": lambda term, **kw: optim.PRIDU(term, maxiter=12, step=0.95, **kw),
    "pridu adaptive": lambda term, **kw: optim.PRIDU(
        term, maxiter=12, step=0.95, adaptive_step=True, **kw
    ),
    "pridu ratio": lambda term, **kw: optim.PRIDU(
        term, maxiter=12, step=0.95, sigma_tau_ratio=3.0, **kw
    ),
}
#: Terms over the image, which every proximal iteration takes, and a term over
#: a transform, which only the primal-dual one is given the transform of.
_ON_THE_IMAGE = {"l1": priors.L1(0.05), "wavelet": priors.Wavelet((-1, -2), 0.02)}
_OVER_A_TRANSFORM = {"laplace": priors.Laplace((-1, -2), 0.02)}


def _solver_cases():
    cases = []
    for solvers, terms in (
        ({**_THRESHOLDING, **_PRIMAL_DUAL}, _ON_THE_IMAGE),
        (_PRIMAL_DUAL, _OVER_A_TRANSFORM),
    ):
        for solver, make in solvers.items():
            for name, term in terms.items():
                cases.append(pytest.param(make, term, id=f"{solver}-{name}"))
    return cases


@pytest.mark.parametrize(("make", "term"), _solver_cases())
@pytest.mark.parametrize("weight", [0.0, 0.1], ids=["no weight", "a quadratic weight"])
def test_the_public_solvers_are_the_library_they_replaced(make, term, weight):
    """`__call__` runs the iteration here; `_in_library` runs BART's."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    solver = make(term, cclambda=weight)

    _solver_agrees(solver, solver(y, A), solver._in_library(y, A))


@pytest.mark.parametrize("solver", [optim.IST, optim.FISTA], ids=["ist", "fista"])
@pytest.mark.parametrize(
    "term",
    [
        priors.FourierL1((-1, -2), 0.05),
        priors.Laplace((-1, -2), 0.02),
        priors.ImaginaryL1(0.05),
        priors.TotalVariation((-1, -2), 0.01),
    ],
    ids=repr,
)
def test_the_thresholding_iterations_refuse_a_term_over_a_transform(solver, term):
    """`pics` hands `iter2_ist` and `iter2_fista` the proximal operators and no
    transforms, so a term over a transform would be thresholded on the image
    itself -- the Fourier coefficients' threshold applied to the pixels -- and
    total variation's proximal operator does not take the image's shape at
    all, which BART asserts.  Both routes refuse before either happens."""
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    with pytest.raises(ValueError, match="penalizes a transform"):
        solver(term, maxiter=3)(y, A)
    with pytest.raises(ValueError, match="penalizes a transform"):
        solver(term, maxiter=3)._in_library(y, A)
    block = optim.FISTABlock(term) if solver is optim.FISTA else optim.ISTBlock(term)
    with pytest.raises(ValueError, match="penalizes a transform"):
        block.start(y, A)


def test_a_fourier_l1_term_thresholds_the_fourier_coefficients():
    """Denoising, ``A = I``: the minimizer of ``1/2 ||x - y||^2 + lambda ||F x||_1``
    is ``y`` with its Fourier coefficients soft-thresholded.  BART's ``F`` is
    the unnormalized transform, so on unitary coefficients the threshold is
    ``lambda sqrt(N)``.  ADMM, which is given the transform, reaches it."""
    torch.manual_seed(0)
    n = 16
    shape = (1, n, n)
    F = linop.FFT(shape, axes=(-1, -2))
    coefficients = 0.05 * _rand(*shape)
    coefficients[0, 3, 4] += 5.0
    coefficients[0, 10, 2] += 3.0j
    y = F.adjoint(coefficients)
    weight = 0.01

    got = optim.ADMM(priors.FourierL1((-1, -2), weight), maxiter=1000)(y, linop.Identity(shape))

    threshold = weight * n
    c = F(y)
    shrunk = torch.where(c.abs() > threshold, (1 - threshold / c.abs()) * c, torch.zeros(()))
    expected = F.adjoint(shrunk)
    assert torch.linalg.vector_norm(got - expected) < 1e-4 * torch.linalg.vector_norm(y)


def test_the_proximal_iterations_take_one_term():
    """`iter2_ist` and `iter2_fista` assert it, and an assertion is the
    process; this is the same refusal, said in Python."""
    with pytest.raises(ValueError, match="exactly one term"):
        optim.FISTA([priors.L1(0.05), priors.L1(0.02)], maxiter=5)
    with pytest.raises(ValueError, match="exactly one term"):
        optim.IST(maxiter=5)


def test_a_term_kept_across_solves_is_rewound_as_the_tool_rewinds_it():
    """A wavelet threshold's cycle spinning comes from a generator of its own,
    seeded when BART makes the operator.  The tool builds a fresh operator per
    run; a term here is kept, so every solve puts the generator back."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = priors.Wavelet((-1, -2), 0.02)

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
    solver = optim.FISTA(priors.L1(0.05), maxiter=8, step=0.7)
    assert torch.equal(solver(zero, A), solver._in_library(zero, A))


# --- the whole of BART's ADMM ---------------------------------------------------
#
# `admm.c` has more in it than `pics` has flags for.  What `italgo_config` can
# be told -- the penalty adaptation, the residual balancing, the fast mode --
# is checked against BART's own loop below.  What it cannot is reachable only
# from the iteration written here, and is checked by what it changes.


@pytest.mark.parametrize("dynamic_rho", [False, True], ids=["fixed rho", "dynamic rho"])
@pytest.mark.parametrize("dynamic_tau", [False, True], ids=["fixed tau", "dynamic tau"])
@pytest.mark.parametrize("relative_norm", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("term", [priors.L1(0.05), priors.TotalVariation((-1, -2), 0.01)], ids=repr)
def test_the_penalty_adaptation_is_barts(dynamic_rho, dynamic_tau, relative_norm, term):
    """Boyd's moving penalty, and the residual balancing of Wohlberg (2017)
    that `dynamic_tau` and `relative_norm` together make of it.  BART has all
    of it; the wrapper had none of it until now."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    solver = optim.ADMM(
        term,
        maxiter=12,
        cg_maxiter=4,
        rho=0.5,
        dynamic_rho=dynamic_rho,
        dynamic_tau=dynamic_tau,
        relative_norm=relative_norm,
    )

    assert torch.equal(solver(y, A), solver._in_library(y, A))


def test_fast_mode_skips_the_residuals_and_is_barts():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    solver = optim.ADMM(priors.L1(0.05), maxiter=12, cg_maxiter=4, fast=True)
    assert torch.equal(solver(y, A), solver._in_library(y, A))


def test_what_bart_asserts_apart_is_refused_here():
    """An assertion in the library takes the process; these come back."""
    with pytest.raises(ValueError, match="hogwild or a dynamic rho"):
        optim.ADMM(priors.L1(0.05), hogwild=True, dynamic_rho=True)
    with pytest.raises(ValueError, match="fast mode does not compute"):
        optim.ADMM(priors.L1(0.05), fast=True, dynamic_rho=True)


@pytest.mark.parametrize(
    "settings",
    [
        {"alpha": 1.0},
        {"mu": 1.5, "dynamic_rho": True},
        {"tau_max": 1.05, "dynamic_rho": True, "dynamic_tau": True, "relative_norm": True},
        {"abstol": 1.0},
        {"reltol": 1.0},
        {"cg_maxiter_first": 12},
    ],
    ids=["over-relaxation", "mu", "tau max", "abstol", "reltol", "a first-step budget"],
)
def test_what_the_tool_cannot_be_told_is_refused_by_its_loop(settings):
    """`italgo_config` builds the configuration `pics` builds.  Asking for
    something it has no way to pass and then for BART's own loop is refused
    rather than quietly dropped."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    solver = optim.ADMM(priors.L1(0.05), maxiter=12, cg_maxiter=2, **settings)

    with pytest.raises(ValueError, match="cannot be set on BART's own loop"):
        solver._in_library(y, A)

    # The same solver but for the setting the tool cannot be told, so what is
    # compared is that setting and nothing else.
    reachable = {k: v for k, v in settings.items() if k not in optim.ADMM._beyond_the_tool}
    plain = optim.ADMM(priors.L1(0.05), maxiter=12, cg_maxiter=2, **reachable)
    assert not torch.equal(solver(y, A), plain(y, A)), f"{settings} changed nothing"


def test_a_first_step_budget_is_rieslings_iters0():
    """The one thing riesling's ADMM has that BART's does not: more inner
    iterations on the first outer step, where there is no warm start to build
    on, and fewer after."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))

    applications = []
    P = linop.LinearOperator.from_callbacks(
        (1, n, n),
        (1, n, n),
        A.forward,
        A.adjoint,
        lambda v: (applications.append(1), A.normal(v))[1],
    )

    # One outer step, so what is counted is that step's inner solve.
    optim.ADMM(priors.L1(0.05), maxiter=1, cg_maxiter=1, cg_maxiter_first=8)(y, P)
    generous = len(applications)

    applications.clear()
    optim.ADMM(priors.L1(0.05), maxiter=1, cg_maxiter=1)(y, P)
    assert generous > len(applications)

    # And after the first step it is the ordinary budget again, so the whole
    # run spends its budget sooner rather than later.
    applications.clear()
    optim.ADMM(priors.L1(0.05), maxiter=40, cg_maxiter=1, cg_maxiter_first=8)(y, P)
    front_loaded = len(applications)
    applications.clear()
    optim.ADMM(priors.L1(0.05), maxiter=40, cg_maxiter=1)(y, P)
    assert front_loaded < len(applications)


def test_a_bias_is_the_offset_bart_solves_with():
    """`f_j(G_j x - b_j)`.  BART's ADMM takes one per term; `italgo_config`
    gives no way to pass it, so it arrives through the iteration here."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = priors.L1(0.05)
    bias = _rand(*SHAPE)

    pulled = optim.ADMM(term, maxiter=12, cg_maxiter=4, biases=[bias])(y, A)
    plain = optim.ADMM(term, maxiter=12, cg_maxiter=4)(y, A)
    assert not torch.equal(pulled, plain)
    assert (pulled - bias).abs().sum() < (plain - bias).abs().sum()

    with pytest.raises(ValueError, match="one bias per term"):
        optim.ADMM([term, term], biases=[bias])


# --- the primal-dual split -----------------------------------------------------


def test_a_term_whose_transform_is_the_identity_becomes_the_primal_step():
    """Not a question about shapes: the Laplace term's transform has the
    image's shape and is a convolution, which is why BART asks the operator."""
    assert priors.L1(0.05).transform_is_identity(SHAPE)
    assert priors.Wavelet((-1, -2), 0.02).transform_is_identity(SHAPE)
    assert not priors.Laplace((-1, -2), 0.02).transform_is_identity(SHAPE)
    assert not priors.TotalVariation((-1, -2), 0.02).transform_is_identity(SHAPE)

    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    state = optim.PRIDUBlock([priors.L1(0.05), priors.TotalVariation((-1, -2), 0.01)]).start(y, A)
    assert state.primal and 1 == len(state.duals)

    state = optim.PRIDUBlock([priors.TotalVariation((-1, -2), 0.01), priors.L1(0.05)]).start(y, A)
    assert not state.primal and 2 == len(state.duals)


@pytest.mark.parametrize("steps", [1, 4, 12])
def test_several_terms_split_the_way_the_library_splits_them(steps):
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    terms = [priors.L1(0.05), priors.TotalVariation((-1, -2), 0.01)]
    solver = optim.PRIDU(terms, maxiter=steps, step=0.95)

    _pridu_agrees(solver(y, A), solver._in_library(y, A))


@pytest.mark.parametrize(
    "term",
    [priors.TotalVariation((-1, -2), 0.01), priors.Laplace((-1, -2), 0.02)],
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

    _pridu_agrees(solver(y, A), solver._in_library(y, A))


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
    assert maxeigen(A, priors.TotalVariation((-1, -2), 0.01)) > 8.0


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

    solver = optim.FISTA(priors.L1(0.05), maxiter=12, step=0.7, eigen=True)
    assert not torch.equal(solver._in_library(y, A), solver._in_library(y, A))


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
    term = priors.L1(0.05)

    ours = make(term)(y, A)
    torch.testing.assert_close(ours, make(term)._in_library(y, A), rtol=1e-4, atol=1e-6)

    # And it is the estimate doing it: a step four times larger is a
    # different answer.
    assert not torch.equal(ours, optim.FISTA(term, maxiter=12, step=0.7)(y, A))


# --- the encoding keeps its own normal ----------------------------------------


def test_the_gradient_goes_through_the_encodings_own_normal():
    """`A^H A x - A^H y` rather than `A^H (A x - y)`.

    For a Toeplitz encoding the first is a convolution with a point spread
    function and the second a transform and its adjoint -- a measurably
    different answer, and not what the encoding was built for.
    """
    import bartorch.tools as bt

    n, coils = 16, 4
    torch.manual_seed(0)
    maps = _rand(coils, 1, n, n)
    traj = bt.traj(x=n, y=24, r=True)
    A = linop.NoncartesianSense(maps, (coils, n, n), traj=traj, toeplitz=True)
    y = _rand(*A.oshape)

    calls = {"normal": 0, "forward": 0}
    normal, forward = A.normal, A.forward

    def counting_normal(v, out=None):
        calls["normal"] += 1
        return normal(v, out)

    def counting_forward(v, out=None):
        calls["forward"] += 1
        return forward(v, out)

    block = optim.FISTABlock(priors.L1(0.01), step=0.5)
    state = block.start(y, A)
    A.normal, A.forward = counting_normal, counting_forward
    try:
        for _ in range(2):
            state = block(state, A)
    finally:
        del A.normal, A.forward
    assert calls == {"normal": 2, "forward": 0}

    x = _rand(*A.ishape)
    assert (A.normal(x) - A.adjoint(A(x))).abs().max() / A.normal(x).abs().max() > 1e-5


def test_the_adjoint_data_is_computed_once_per_solve():
    """It does not change during one, and an extra adjoint a step is a transform a step."""
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))

    calls = []
    plain = A.adjoint

    def counting(v, out=None):
        calls.append(1)
        return plain(v, out)

    A.adjoint = counting
    try:
        optim.FISTA(priors.L1(0.05), maxiter=5, step=0.7)(y, A)
    finally:
        del A.adjoint
    assert len(calls) == 1


# --- alternating directions ---------------------------------------------------
#
# BART's ADMM solves `min 0.5||Ax-y||^2 + sum_j f_j(G_j x - b_j)`, which is the
# one iteration here that takes several terms, each with its own transform and
# its own bias.  Its x-update is conjugate gradients on
# `A^H A + rho sum_j G_j^H G_j`, warm-started, which is why it stays BART's
# even though the outer loop is not.


def _admm_steps(A, y, terms, steps, *, biases=None, rho=0.5, cg=10, **settings):
    block = optim.ADMMBlock(terms, biases=biases, rho=rho, cg_maxiter=cg, **settings)
    state = block.start(y, A)
    for _ in range(steps):
        state = block(state, A)
    return state.x


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 8])
def test_admm_is_barts_admm(steps):
    """On a problem whose inner solve converges in one iteration, BART's
    budget and the outer step count are the same number, so the two can be
    held against each other directly."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = priors.L1(0.05)

    ours = _admm_steps(A, y, [term], steps)
    theirs = optim.ADMM(term, maxiter=steps, cg_maxiter=10, rho=0.5)._in_library(y, A)
    assert torch.equal(ours, theirs)


def test_one_admm_step_is_barts_step_on_a_harder_problem():
    """Where the inner solve takes several iterations, BART's `maxiter` stops
    counting outer steps -- but the step itself is still the same one."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    term = priors.L1(0.05)

    ours = _admm_steps(A, y, [term], 1)
    theirs = optim.ADMM(term, maxiter=1, cg_maxiter=10, rho=0.5)._in_library(y, A)
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
    P = linop.LinearOperator.from_callbacks(
        (1, n, n),
        (1, n, n),
        base.forward,
        base.adjoint,
        lambda v: (applications.append(1), base.normal(v))[1],
    )
    optim.ADMM(priors.L1(0.05), maxiter=30, cg_maxiter=10, rho=0.5)._in_library(y, P)

    # Thirty outer steps at ten inner iterations would be hundreds.
    assert 30 < len(applications) < 100


def test_total_variation_goes_through_a_gradient_an_operator_cannot_hold():
    """The term ADMM is for, and the one whose transform is rank seventeen.

    `transform()` refuses it; `apply_transform` is what the iteration uses,
    and the dot test says the pair it applies really are adjoint.
    """
    torch.manual_seed(0)
    term = priors.TotalVariation((-1, -2), 0.05)
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
    terms = [priors.L1(0.05), priors.Wavelet((-1, -2), 0.02)]
    out = _admm_steps(A, y, terms, 4)
    assert out.shape == A.ishape
    assert torch.isfinite(out.abs()).all()


# --- the public solver runs this loop ------------------------------------------
#
# `optim.ADMM` no longer hands the whole solve to `lsqr2`: it drives
# `ADMMBlock` step by step.
# What follows is the claim that buys: the two paths answer with the same bits.


@pytest.mark.parametrize("budget", [1, 3, 7, 12, 30])
def test_the_public_solver_is_the_library_it_replaced(budget):
    """`ADMM.__call__` loops the block; `ADMM._in_library` runs BART's."""
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    solver = optim.ADMM(priors.L1(0.05), maxiter=budget, cg_maxiter=4, rho=0.5)

    assert torch.equal(solver(y, A), solver._in_library(y, A))


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
    terms = [priors.L1(0.05), priors.L1(0.02)]
    solver = optim.ADMM(terms, maxiter=8, cg_maxiter=4, rho=0.5)

    assert torch.equal(solver(y, A), solver._in_library(y, A))


def test_the_loop_crosses_into_python_where_the_library_would_not():
    """The point of the switch, and the cost of it: the step is visible."""
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))

    seen = []
    term = priors.L1(0.05)
    prox_of = term.prox
    term.prox = lambda *a, **kw: (seen.append(1), prox_of(*a, **kw))[1]

    optim.ADMM(term, maxiter=5, cg_maxiter=4)(y, A)
    assert seen, "the threshold was applied without passing through Python"


def test_a_bias_pulls_the_split_variable_towards_it():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = priors.L1(0.05)

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
        _admm_steps(A, y, [priors.L1(0.01)], 2, cg=3)
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


def _pridu_steps(A, y, steps, *, terms=(), primal=None, step=0.95, ratio=1.0, **settings):
    block = optim.PRIDUBlock(
        ([primal] if primal is not None else []) + list(terms),
        step=step,
        sigma_tau_ratio=ratio,
        **settings,
    )
    state = block.start(y, A)
    for _ in range(steps):
        state = block(state, A)
        if state.done:
            break
    return state.x


# --- one multiply-add BART's compiler is free to fuse --------------------------
#
# `vecops.c` has one kernel behind `axpy`, `xpay` and `axpbz`:
#
#     dst[i] = a1 * src1[i] + a2 * src2[i];
#
# and clang contracts the first product into the add where the hardware has a
# fused multiply-add -- arm64 does, the x86-64 baseline does not.  A fused
# multiply-add does not round the product, so on arm64 that is one rounding
# where this package computes two, and torch has no way to fuse across two
# kernels.
#
# It matters for exactly one of the four iterations.  `axpy` passes `a1 = 1.`,
# and fusing an exact product changes nothing, so every iteration whose
# updates are axpys -- IST, FISTA, ADMM -- is the same bits on either
# platform.  `chambolle_pock` is the one that reaches for `xpay` and `axpbz`
# with two real coefficients, in the data term's resolvent, and there the
# fused and unfused readings part.
#
# So the primal-dual tests ask the library which arithmetic it was compiled
# with, rather than assuming, and hold the iteration to the bit where it can
# be held to the bit.


def _f32(v):
    return float(np.float32(v))


def _fused(a, x, y):
    """``a * x + y`` as a fused multiply-add: the product is not rounded.

    Single-precision inputs make the double exact, so rounding the double
    once is what the instruction does.
    """
    parts = torch.view_as_real(x).double() * float(a) + torch.view_as_real(y).double()
    return parts.float().view(torch.complex64).squeeze(-1)


def _replay(A, y, term, steps, *, fused):
    """`chambolle_pock` for a single primal term, with and without the fusion.

    Written out rather than driven through `PRIDUBlock` because the point
    is to vary the arithmetic inside the step.  `_library_fuses` checks the
    unfused reading against the iteration itself before believing either.

    Only used at step counts too small for the tolerance to stop the
    iteration, so it carries no residual test.
    """
    op = A._bart()
    adjoint = op.adjoint(y)
    sigma = tau = _f32(math.sqrt(0.95))
    keep, pull = _f32(1.0 / (1.0 + sigma)), _f32(-1.0 * sigma / (1.0 + sigma))

    x = torch.zeros(*op.ishape, dtype=torch.complex64)
    avg, dual = x, x
    for _ in range(steps):
        moved = op.normal(avg)
        if fused:
            # `xpay(sigma, Ahu_old, Ahu)` and
            # `axpbz(Ahu_new, keep, Ahu_old, pull, xadj)`, each with its first
            # product folded into the add.
            step = _fused(sigma, moved, dual)
            dual = _fused(keep, step, pull * adjoint)
        else:
            step = sigma * moved + dual
            dual = keep * step + pull * adjoint

        previous = x
        # `axpy(x, -tau, Ahu)` is `1. * x[i] + (-tau) * Ahu[i]`, and fusing an
        # exact product changes nothing, so this one reads the same either way.
        x = x - tau * dual
        x = term.prox(x, tau, image_shape=op.ishape)
        avg = 2.0 * x - previous
    return x


@functools.lru_cache(maxsize=1)
def _library_fuses() -> bool:
    """Whether this build of BART folds those products into their adds.

    Asked of the library rather than of `platform.machine()`: it is the
    compiler's choice, and the answer is whichever reading reproduces a solve.
    """
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    y = A(_rand(*SHAPE))
    term = priors.L1(0.05)
    theirs = optim.PRIDU(term, maxiter=2, step=0.95)._in_library(y, A)

    plain = _replay(A, y, term, 2, fused=False)
    assert torch.equal(plain, _pridu_steps(A, y, 2, primal=term)), (
        "the replay does not reproduce the iteration it is meant to vary, so "
        "nothing it says about the library can be trusted"
    )
    if torch.equal(plain, theirs):
        return False

    assert torch.equal(_replay(A, y, term, 2, fused=True), theirs), (
        "the library agrees with neither reading of its own multiply-add, so "
        "the primal-dual step differs from this one for some other reason"
    )
    return True


def _pridu_agrees(ours, theirs):
    """To the bit, or -- where the library fused -- as close as that allows."""
    if not _library_fuses():
        assert torch.equal(ours, theirs)
    else:
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


def test_the_build_does_not_fold_a_multiply_into_an_add():
    """`CMakeLists.txt` pins `-ffp-contract=off`, and this is what says so.

    Without it clang folds the product into the add on any target that has a
    fused multiply-add, and BART then computes something slightly different
    from what it computes elsewhere -- which no amount of rearranging on this
    side can follow, because torch cannot fold across two kernels.  Measured
    on arm64 before the flag: the primal-dual solver a few times 1e-5 from the
    library over twelve steps, and every solve carrying a quadratic weight
    adrift as well.

    If this fails, the flag has been lost from the build rather than anything
    being wrong with the iterations.  `_pridu_agrees` keeps the rest of the
    suite readable in that case rather than failing everywhere at once.
    """
    assert not _library_fuses(), (
        "this build folds a multiply into an add, so it is not the arithmetic "
        "the rest of the suite is checked against; `-ffp-contract=off` has "
        "gone missing from CMakeLists.txt"
    )


def test_the_fusion_is_in_the_resolvent_and_not_in_the_steps():
    """Which operations it reaches, measured rather than reasoned about.

    This is what says IST, FISTA and ADMM are the same bits on every platform
    and the primal-dual iteration is not: the difference is whether the
    kernel's first coefficient is one.
    """
    torch.manual_seed(0)
    x, v = _rand(*SHAPE), _rand(*SHAPE)

    # `axpy`: `a1 = 1.`, an exact product, so folding it in changes nothing.
    assert torch.equal(_fused(1.0, x, 0.3 * v), 1.0 * x + 0.3 * v)

    # `xpay` and `axpbz` carry a coefficient on the first vector too, and
    # there the two readings are different numbers.
    assert not torch.equal(_fused(0.7, x, 0.3 * v), 0.7 * x + 0.3 * v)


@pytest.mark.parametrize("steps", [1, 2, 5, 10, 25, 60])
def test_pridu_is_barts_pridu(problem, steps):
    """The term's transform is the identity, so it is the primal step."""
    A, y = problem
    term = priors.L1(0.05)
    ours = _pridu_steps(A, y, steps, primal=term)
    _pridu_agrees(ours, optim.PRIDU(term, maxiter=steps, step=0.95)._in_library(y, A))


@pytest.mark.parametrize("steps", [1, 3, 8])
def test_a_term_with_a_transform_becomes_a_dual(problem, steps):
    """Total variation's is a gradient, so it cannot be the primal step and
    `prox2` falls back to the identity, as `prox_zero_create` is."""
    A, y = problem
    term = priors.TotalVariation((-1, -2), 0.05)
    ours = _pridu_steps(A, y, steps, terms=[term])
    _pridu_agrees(ours, optim.PRIDU(term, maxiter=steps, step=0.95)._in_library(y, A))


@pytest.mark.parametrize("steps", [1, 3, 8, 20])
def test_hogwild_is_a_decay_here_rather_than_a_halving(problem, steps):
    """0.95 a step, and `(float)pow(decay, i)` from a float32 `decay` -- taken
    in double and rounded once, which is not the same as taking it in one."""
    A, y = problem
    term = priors.L1(0.05)
    ours = _pridu_steps(A, y, steps, primal=term, hogwild=True)
    _pridu_agrees(ours, optim.PRIDU(term, maxiter=steps, step=0.95, hogwild=True)._in_library(y, A))


@pytest.mark.parametrize("steps", [1, 3, 8])
def test_the_adaptive_step_is_barts(problem, steps):
    A, y = problem
    term = priors.L1(0.05)
    ours = _pridu_steps(A, y, steps, primal=term, adaptive_step=True)
    _pridu_agrees(
        ours, optim.PRIDU(term, maxiter=steps, step=0.95, adaptive_step=True)._in_library(y, A)
    )


@pytest.mark.parametrize("ratio", [0.5, 2.0])
def test_the_step_ratio_splits_sigma_and_tau_the_way_pics_does(problem, ratio):
    A, y = problem
    term = priors.L1(0.05)
    ours = _pridu_steps(A, y, 10, primal=term, ratio=ratio)
    _pridu_agrees(
        ours, optim.PRIDU(term, maxiter=10, step=0.95, sigma_tau_ratio=ratio)._in_library(y, A)
    )
