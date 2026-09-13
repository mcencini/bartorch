"""The solvers as networks: unrolled, and at a fixed point.

An iteration written out here is a ``deepinv.optim.optim_iterators.
OptimIterator``, so it goes into ``BaseOptim`` with ``unfold=True`` for a
network of K steps or ``DEQ=True`` for a deep-equilibrium model.
:meth:`unrolled` and :meth:`fixed_point` are that wiring, and the claim they
make is a strong one: with nothing trainable in it, a network answers with the
solver's bits, batch or no batch.  What it is then is BART's iteration with a
denoiser in the threshold's place, rather than an architecture that resembles
one.

Two solvers refuse a fixed point and say why.  FISTA's momentum depends on the
iteration number, so its step is a different map every time; an
alternating-direction step's fixed point is in ``(x, z, u)`` rather than in
the image, and its x-update reaches the previous image only through a warm
start, which carries no gradient.
"""

import pytest
import torch

import bartorch
from bartorch import linop, optim, prox
from bartorch.optim.iterators import AsTerm

SHAPE = (1, 8, 8)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    return A, A(_rand(*SHAPE)), bartorch.to_deepinv(A)


class _Scale(torch.nn.Module):
    """A denoiser with one weight in it, which is all a gradient needs."""

    def __init__(self, weight: float = 0.8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def prox(self, x, *args, gamma=1.0, **kwargs):
        return self.weight * x


_SOLVERS = {
    "ist": lambda term: optim.IST(term, maxiter=5, step=0.7),
    "fista": lambda term: optim.FISTA(term, maxiter=5, step=0.7),
    "admm": lambda term: optim.ADMM(term, maxiter=6, cg_maxiter=2, rho=0.5),
    "pridu": lambda term: optim.PRIDU(term, maxiter=5, step=0.95),
}
_TRAINS = {"ist": ["stepsize"], "fista": ["stepsize"], "admm": ["rho"], "pridu": ["sigma", "tau"]}


# --- with nothing trainable, it is the solver ---------------------------------


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_a_network_with_nothing_to_learn_is_the_solver_to_the_bit(problem, name):
    A, y, physics = problem
    direct = _SOLVERS[name](prox.L1(0.05))(y, A)
    network = _SOLVERS[name](prox.L1(0.05)).unrolled(SHAPE)
    with torch.no_grad():
        through = network(y[None], physics)
    assert torch.equal(direct, through[0])


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_a_batch_is_the_same_answers_side_by_side(problem, name):
    # BART has no batch axis, so a batch is the solver run once per item --
    # including the conjugate gradients inside an alternating-direction step,
    # which would otherwise stop on the whole stack's residual and let the
    # items steer each other.
    A, _, physics = problem
    data = torch.stack([A(_rand(*SHAPE)) for _ in range(3)])
    network = _SOLVERS[name](prox.L1(0.05)).unrolled(SHAPE)
    with torch.no_grad():
        batched = network(data, physics)
    for i in range(3):
        assert torch.equal(batched[i], _SOLVERS[name](prox.L1(0.05))(data[i], A))


def test_a_network_starts_where_bart_starts(problem):
    # `deepinv` starts an optimizer at `A^H y`; BART starts at zero, and that
    # is what makes the answers above the same.  The other start is one
    # keyword away, because a network about to be trained usually wants it.
    A, y, physics = problem
    solver = optim.IST(prox.L1(0.05), maxiter=5, step=0.7)
    with torch.no_grad():
        ours = solver.unrolled(SHAPE)(y[None], physics)
        theirs = solver.unrolled(SHAPE, custom_init=None)(y[None], physics)
    assert torch.equal(ours[0], solver(y, A))
    assert not torch.equal(ours, theirs)


# --- what a network learns ----------------------------------------------------


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_the_gradient_reaches_the_denoiser_and_the_learned_parameters(problem, name):
    _, y, physics = problem
    network = _SOLVERS[name](AsTerm(_Scale())).unrolled(SHAPE, trainable=_TRAINS[name])
    network(y[None], physics).abs().square().sum().backward()

    named = dict(network.named_parameters())
    trained = [n for n, p in named.items() if p.grad is not None and torch.any(p.grad != 0)]
    assert any("weight" in n for n in trained), f"the denoiser was not reached: {list(named)}"
    for parameter in _TRAINS[name]:
        assert any(parameter in n for n in trained), f"{parameter} was not reached"


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_the_denoisers_weights_are_the_networks_weights(problem, name):
    """``net.parameters()`` has to reach them or there is nothing to train."""
    denoiser = _Scale()
    network = _SOLVERS[name](AsTerm(denoiser)).unrolled(SHAPE)
    assert any(p is denoiser.weight for p in network.parameters())


def test_a_step_size_that_is_learned_is_no_longer_the_libraries(problem):
    # A learned parameter is a tensor, so the scalars are worked out in single
    # precision throughout rather than in a double rounded at the end.  The
    # answer moves, and saying so is the point of the test.
    _, y, physics = problem
    solver = optim.IST(prox.frozen(prox.L1(0.05)), maxiter=5, step=0.7)
    plain = solver.unrolled(SHAPE)
    learned = solver.unrolled(SHAPE, trainable=["stepsize"])
    with torch.no_grad():
        assert torch.allclose(plain(y[None], physics), learned(y[None], physics), atol=1e-5)


def test_a_parameter_that_is_not_one_says_so(problem):
    with pytest.raises(ValueError, match="no parameter called 'rho' to learn"):
        optim.IST(prox.L1(0.05), maxiter=4).unrolled(SHAPE, trainable=["rho"])


def test_a_solver_that_runs_in_the_library_has_nothing_to_unroll():
    with pytest.raises(TypeError, match="no iteration written out here"):
        optim.CG(maxiter=4).unrolled(SHAPE)


def test_a_step_from_a_power_iteration_needs_an_encoding_a_network_has_not_got():
    with pytest.raises(ValueError, match="power iteration over the encoding"):
        optim.FISTA(prox.L1(0.05), maxiter=4, eigen=True).unrolled(SHAPE)
    with pytest.raises(ValueError, match="power iteration over the encoding"):
        optim.PRIDU(prox.L1(0.05), maxiter=4, eigen=True).unrolled(SHAPE)


# --- at a fixed point ---------------------------------------------------------


@pytest.mark.parametrize("name", ["ist", "pridu"])
def test_the_fixed_point_is_one(problem, name):
    """What comes back is a point the step leaves where it is."""
    A, y, physics = problem
    solver = {
        "ist": optim.IST(prox.L1(0.02), maxiter=300, step=0.7),
        "pridu": optim.PRIDU(prox.L1(0.02), maxiter=300, step=0.95),
    }[name]
    with torch.no_grad():
        point = solver.fixed_point(SHAPE)(y[None], physics)[0]
        moved = {
            "ist": optim.IST(prox.L1(0.02), maxiter=1, step=0.7),
            "pridu": optim.PRIDU(prox.L1(0.02), maxiter=1, step=0.95),
        }[name]
        again = moved(y, A, point)
    assert (again - point).abs().max() < 1e-2 * point.abs().max()


def test_a_fixed_point_model_trains_through_the_point_it_found(problem):
    _, y, physics = problem
    network = optim.IST(AsTerm(_Scale(0.5)), maxiter=40, step=0.7).fixed_point(SHAPE)
    network(y[None], physics).abs().square().sum().backward()
    weights = [p for n, p in network.named_parameters() if "weight" in n]
    assert weights and all(p.grad is not None for p in weights)


def test_fista_says_why_it_has_no_fixed_point():
    with pytest.raises(TypeError, match="no fixed point"):
        optim.FISTA(prox.L1(0.05), maxiter=8).fixed_point(SHAPE)


def test_alternating_directions_says_where_its_fixed_point_is_instead():
    with pytest.raises(TypeError, match=r"fixed point is in \(x, z, u\)"):
        optim.ADMM(prox.L1(0.05), maxiter=8).fixed_point(SHAPE)
