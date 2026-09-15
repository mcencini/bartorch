"""The solvers as functions, and a denoiser standing where a term goes.

Two claims.  That `optim.fista(y, A, term, ...)` is `optim.FISTA(term,
...)(y, A)` and nothing else -- so the one-line form costs nothing and the
configured form stays the one to keep.  And that a ``deepinv`` prior goes
wherever a :mod:`bartorch.priors` term goes, which is what makes a plug-and-play
reconstruction BART's iteration with the threshold replaced rather than a
second implementation of it.
"""

import pytest
import torch

from bartorch import linop, optim, priors

SHAPE = (1, 8, 8)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    return A, A(_rand(*SHAPE))


# --- the same thing said in one expression ------------------------------------


@pytest.mark.parametrize(
    "call,build",
    [
        (
            lambda y, A, t: optim.ist(y, A, t, maxiter=8, step=0.7),
            lambda t: optim.IST(t, maxiter=8, step=0.7),
        ),
        (
            lambda y, A, t: optim.fista(y, A, t, maxiter=8, step=0.7),
            lambda t: optim.FISTA(t, maxiter=8, step=0.7),
        ),
        (
            lambda y, A, t: optim.admm(y, A, t, maxiter=8, cg_maxiter=4),
            lambda t: optim.ADMM(t, maxiter=8, cg_maxiter=4),
        ),
        (
            lambda y, A, t: optim.pridu(y, A, t, maxiter=8, step=0.95),
            lambda t: optim.PRIDU(t, maxiter=8, step=0.95),
        ),
    ],
    ids=["ist", "fista", "admm", "pridu"],
)
def test_the_function_is_the_class_configured_and_called(call, build, problem):
    A, y = problem
    term = priors.L1(0.05)
    assert torch.equal(call(y, A, term), build(term)(y, A))


def test_conjugate_gradients_too(problem):
    A, y = problem
    assert torch.equal(optim.cg(y, A, 0.1, maxiter=8), optim.CG(0.1, maxiter=8)(y, A))


def test_a_warm_start_goes_through(problem):
    A, y = problem
    term = priors.L1(0.05)
    x0 = _rand(*SHAPE)
    assert torch.equal(
        optim.fista(y, A, term, maxiter=8, step=0.7, x0=x0),
        optim.FISTA(term, maxiter=8, step=0.7)(y, A, x0),
    )
    assert not torch.equal(
        optim.fista(y, A, term, maxiter=8, step=0.7, x0=x0),
        optim.fista(y, A, term, maxiter=8, step=0.7),
    )


def test_several_terms_are_taken_as_a_list(problem):
    A, y = problem
    terms = [priors.L1(0.05), priors.TotalVariation((-1, -2), 0.01)]
    assert torch.equal(
        optim.admm(y, A, terms, maxiter=8, cg_maxiter=4),
        optim.ADMM(terms, maxiter=8, cg_maxiter=4)(y, A),
    )


# --- a denoiser where a term goes ---------------------------------------------


def _denoiser(kernel_size: int = 3):
    from deepinv.models import MedianFilter, to_complex_denoiser

    # An image here is complex and most denoisers are not; `median_out` is not
    # implemented for complex tensors at all, so this is the wrapper's job.
    return to_complex_denoiser(MedianFilter(kernel_size=kernel_size))


@pytest.mark.parametrize(
    "solve",
    [
        lambda y, A, p: optim.fista(y, A, p, maxiter=6, step=0.7, g_param=0.05),
        lambda y, A, p: optim.ist(y, A, p, maxiter=6, step=0.7, g_param=0.05),
        lambda y, A, p: optim.admm(y, A, p, maxiter=6, cg_maxiter=3, g_param=0.05),
        lambda y, A, p: optim.pridu(y, A, p, maxiter=6, step=0.95, g_param=0.05),
    ],
    ids=["fista", "ist", "admm", "pridu"],
)
def test_a_denoiser_stands_where_a_term_would(solve, problem):
    A, y = problem
    got = solve(y, A, _denoiser())
    assert got.shape == A.ishape
    assert torch.isfinite(got.abs()).all()
    assert not torch.equal(got, torch.zeros_like(got))


def test_a_deepinv_prior_goes_through_as_it_stands(problem):
    from deepinv.optim import PnP

    A, y = problem
    prior = PnP(denoiser=_denoiser())
    bare = optim.fista(y, A, prior, maxiter=6, step=0.7, g_param=0.05)
    wrapped = optim.fista(y, A, _denoiser(), maxiter=6, step=0.7, g_param=0.05)
    assert torch.equal(bare, wrapped)


def test_a_denoiser_and_a_term_split_apart_in_the_same_solve(problem):
    """Which is what the alternating-direction solver is for: each term gets
    its own split variable, whoever wrote it."""
    A, y = problem
    got = optim.admm(
        y,
        A,
        [_denoiser(), priors.TotalVariation((-1, -2), 0.01)],
        maxiter=6,
        cg_maxiter=3,
        g_param=0.05,
    )
    assert got.shape == A.ishape
    assert torch.isfinite(got.abs()).all()


def test_the_denoiser_is_what_changes_the_answer(problem):
    """A median filter takes no noise level, so it is its window that says how
    much it smooths.  Which is the point: the prior is whatever was handed
    over, and the iteration does not look inside it."""
    A, y = problem
    wide = optim.fista(y, A, _denoiser(7), maxiter=6, step=0.7, g_param=0.05)
    narrow = optim.fista(y, A, _denoiser(3), maxiter=6, step=0.7, g_param=0.05)
    assert not torch.equal(wide, narrow)


# --- and what it cannot be ----------------------------------------------------


def test_there_is_no_library_route_for_a_denoiser(problem):
    """BART has no way to be handed one, and the refusal says so rather than
    quietly running something else."""
    from bartorch.optim._iterators import AsTerm

    A, y = problem
    solver = optim.FISTA(AsTerm(_denoiser(), 0.05), maxiter=6, step=0.7)
    with pytest.raises(ValueError, match="no library route"):
        solver.in_library(y, A)


def test_a_terms_own_weight_is_not_a_priors_parameter(problem):
    A, y = problem
    with pytest.raises(ValueError, match="carries its weight"):
        optim.fista(y, A, priors.L1(0.05), maxiter=6, g_param=0.05)


def test_something_that_is_neither_is_still_refused(problem):
    A, y = problem
    with pytest.raises(TypeError, match="a deepinv prior, or a denoiser"):
        optim.fista(y, A, object(), maxiter=6)


# --- and what runs underneath it ----------------------------------------------


@pytest.mark.parametrize(
    "call,iteration",
    [
        (lambda y, A, t: optim.ist(y, A, t, maxiter=4, step=0.7), "ISTIteration"),
        (lambda y, A, t: optim.fista(y, A, t, maxiter=4, step=0.7), "FISTAIteration"),
        (lambda y, A, t: optim.admm(y, A, t, maxiter=2, cg_maxiter=4), "ADMMIteration"),
        (lambda y, A, t: optim.pridu(y, A, t, maxiter=4, step=0.95), "PRIDUIteration"),
    ],
    ids=["ist", "fista", "admm", "pridu"],
)
def test_the_function_runs_the_iteration_and_not_a_second_implementation(
    problem, monkeypatch, call, iteration
):
    """The route is function -> solver -> the iteration in
    :mod:`bartorch.optim._iterators`, and nothing in between writes the
    algorithm out again.

    Worth pinning rather than assuming: the whole claim that a plug-and-play
    solve is BART's own iteration with the threshold replaced rests on there
    being exactly one copy of each iteration, and that is the copy the suite
    holds against the library.
    """
    from bartorch.optim import _iterators as iterators

    cls = getattr(iterators, iteration)
    seen = []
    original = cls.forward
    monkeypatch.setattr(
        cls, "forward", lambda self, *a, **kw: seen.append(1) or original(self, *a, **kw)
    )

    A, y = problem
    call(y, A, priors.L1(0.01))
    assert seen, f"{iteration} was never stepped, so something else ran the iteration"


# --- the function layer covers every solver ---------------------------------


def test_every_solver_has_a_function():
    """The function is the ordinary call path, so a solver reachable only as a
    class would send a caller to the handle for no reason."""
    classes = {n for n in optim.__all__ if n[0].isupper()} - {"Tikhonov"}
    functions = {n for n in optim.__all__ if n.islower()} - {"data_scaling", "maxeigen"}
    assert {c.lower() for c in classes} == functions


def test_niht_is_refused_the_way_its_class_is():
    """BART's own iteration cannot run against the operator `lsqr2` hands it;
    see `tests/test_solve.py`."""
    A = linop.FFT((8, 8), axes=(-1, -2))
    y = A(_rand(8, 8))
    term = priors.WaveletNIHT((-1, -2), count=12)
    with pytest.raises(NotImplementedError, match="applies the normal operator in place"):
        optim.niht(y, A, term, maxiter=6)


def test_irgnm_is_its_class():
    A = linop.FFT((4, 4), axes=-1)
    x = _rand(4, 4)
    start = torch.zeros(4, 4, dtype=torch.complex64)
    torch.testing.assert_close(
        optim.irgnm(A(x), A, x0=start.clone(), inner=optim.CG(), iterations=6, alpha=0.01),
        optim.IRGNM(iterations=6, alpha=0.01, inner=optim.CG())(A(x), A, x0=start.clone()),
        rtol=0.0,
        atol=0.0,
    )
