"""The blocks as networks: a stack of steps, frozen or learned.

With nothing learned, a stack of blocks is the solver to the bit, batch or no
batch.  A learned setting or a denoiser in the prior slot is reached by the
gradient, and rho's is the whole derivative, through the inner solve too.
"""

import pytest
import torch
from torch import nn

from bartorch import linop, optim, priors

SHAPE = (1, 8, 8)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    return A, A(_rand(*SHAPE))


class _Scale(nn.Module):
    """A denoiser with one weight in it, which is all a gradient needs."""

    def __init__(self, weight: float = 0.8):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.seen = []

    def forward(self, x, sigma=None):
        self.seen.append(tuple(x.shape))
        return self.weight * x


#: The solver, the block it loops, and how many steps the solver takes here.
_SOLVERS = {
    "ist": (lambda t: optim.IST(t, maxiter=5, step=0.7), lambda t: optim.ISTBlock(t, step=0.7), 5),
    "fista": (
        lambda t: optim.FISTA(t, maxiter=5, step=0.7),
        lambda t: optim.FISTABlock(t, step=0.7),
        5,
    ),
    "admm": (
        lambda t: optim.ADMM(t, maxiter=6, cg_maxiter=2, rho=0.5),
        lambda t: optim.ADMMBlock(t, cg_maxiter=2, rho=0.5),
        6,
    ),
    "pridu": (
        lambda t: optim.PRIDU(t, maxiter=5, step=0.95),
        lambda t: optim.PRIDUBlock(t, step=0.95),
        5,
    ),
}
_LEARNS = {"ist": ["step"], "fista": ["step"], "admm": ["rho"], "pridu": ["sigma", "tau"]}


def _stack(name, term):
    _, make, depth = _SOLVERS[name]
    return nn.ModuleList(make(term) for _ in range(depth))


def _run(stack, y, A):
    state = stack[0].start(y, A)
    for block in stack:
        state = block(state, A)
    return stack[-1].output(state, A)


# --- with nothing learned, it is the solver ------------------------------------


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_a_stack_with_nothing_learned_is_the_solver_to_the_bit(problem, name):
    A, y = problem
    solver = _SOLVERS[name][0](priors.L1(0.05))
    assert torch.equal(_run(_stack(name, priors.L1(0.05)), y, A), solver(y, A))


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_a_batch_is_the_same_answers_side_by_side(problem, name):
    # BART has no batch axis, so a batch is walked item by item -- including
    # the conjugate gradients inside an alternating-direction step, which
    # would otherwise stop on the whole stack's residual.
    A, _ = problem
    data = torch.stack([A(_rand(*SHAPE)) for _ in range(3)])
    batched = _run(_stack(name, priors.L1(0.05)), data, A)
    for i in range(3):
        assert torch.equal(batched[i], _SOLVERS[name][0](priors.L1(0.05))(data[i], A))


def test_a_denoiser_sees_the_batch_whole(problem):
    A, y = problem
    denoiser = _Scale()
    block = optim.FISTABlock(priors.ImplicitPrior(denoiser), step=0.7)
    data = torch.stack([y, y, y])
    block(block.start(data, A), A)
    block(block.start(y, A), A)
    assert denoiser.seen == [(3, *SHAPE), (1, *SHAPE)]


# --- what a stack learns -------------------------------------------------------


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_the_gradient_reaches_the_denoiser_and_every_blocks_settings(problem, name):
    A, y = problem
    denoiser = _Scale()
    stack = _stack(name, priors.ImplicitPrior(denoiser))
    for block in stack:
        for setting in _LEARNS[name]:
            getattr(block, setting).requires_grad_()

    _run(stack, y, A).abs().square().sum().backward()

    assert denoiser.weight.grad is not None and 0 != denoiser.weight.grad
    for i, block in enumerate(stack):
        for setting in _LEARNS[name]:
            grad = getattr(block, setting).grad
            assert grad is not None and 0 != grad, f"block {i}'s {setting} was not reached"


@pytest.mark.parametrize("name", list(_SOLVERS))
def test_the_denoisers_weights_are_the_stacks_weights(name):
    denoiser = _Scale()
    stack = _stack(name, priors.ImplicitPrior(denoiser))
    assert any(p is denoiser.weight for p in stack.parameters())


def test_settings_are_frozen_until_asked(problem):
    block = optim.FISTABlock(priors.L1(0.05), step=0.7)
    assert [] == [p for p in block.parameters() if p.requires_grad]
    assert block.step.requires_grad_().requires_grad


def test_a_learned_step_is_close_to_but_not_the_libraries(problem):
    # A learned setting is a single-precision tensor, so the scalars are
    # worked out in single precision throughout rather than rounded where
    # BART rounds them.
    A, y = problem
    frozen = optim.ISTBlock(priors.frozen(priors.L1(0.05)), step=0.7)
    learned = optim.ISTBlock(priors.frozen(priors.L1(0.05)), step=0.7)
    learned.step.requires_grad_()
    with torch.no_grad():
        torch.testing.assert_close(
            _run([learned] * 5, y, A), _run([frozen] * 5, y, A), rtol=0, atol=1e-5
        )


def test_the_rho_gradient_is_the_one_finite_differences_measure():
    """Through the inner solve as well as the right-hand side.

    A denoiser's prox ignores its step; a BART term's threshold would move
    with ``1 / rho`` with no derivative to say so.
    """
    torch.manual_seed(0)
    n = 8
    diag = torch.linspace(0.2, 1.0, n * n).to(torch.complex64).reshape(1, n, n)
    A = linop.Diagonal(diag, (1, n, n))
    y = A(_rand(1, n, n))
    term = priors.ImplicitPrior(_Scale(0.8))

    def loss(rho):
        block = optim.ADMMBlock(term, rho=0.5, cg_maxiter=60)
        block.rho.data.fill_(float(rho.detach()) if isinstance(rho, torch.Tensor) else rho)
        if isinstance(rho, torch.Tensor):
            block.rho = rho
        state = block.start(y, A)
        for _ in range(3):
            state = block(state, A)
        return state.x.abs().square().sum()

    rho = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    loss(rho).backward()

    h = 1e-3
    measured = (loss(0.5 + h) - loss(0.5 - h)).item() / (2 * h)
    assert abs(rho.grad.item() - measured) <= 2e-2 * abs(measured)


def test_a_step_from_a_power_iteration_is_estimated_at_start(problem):
    from bartorch.optim.linear import maxeigen

    torch.manual_seed(0)
    diag = torch.full(SHAPE, 0.1, dtype=torch.complex64)
    diag[0, 0, 0] = 2.0
    A = linop.Diagonal(diag, SHAPE)
    state = optim.FISTABlock(priors.L1(0.05), eigen=True).start(A(_rand(*SHAPE)), A)
    assert state.divisor == pytest.approx(maxeigen(A), rel=1e-4)


# --- an encoding linearized at a point ----------------------------------------------


_IMAGE, _COILS = (1, 4), (3, 4)
_STATE = 16


def _jacobian(point):
    """``DF(point)`` of ``a * b`` as a dense matrix, differentiable by the point."""
    a, b = point[:4].reshape(_IMAGE), point[4:].reshape(_COILS)
    columns = []
    for at in range(_STATE):
        unit = torch.zeros(_STATE, dtype=torch.complex64)
        unit[at] = 1.0
        da, db = unit[:4].reshape(_IMAGE), unit[4:].reshape(_COILS)
        columns.append((da * b + a * db).reshape(-1))
    return torch.stack(columns, dim=1)


class _Dense(linop.LinearOperator):
    """The same linearization written in torch, which differentiates it by itself."""

    _records = True

    def __init__(self, point):
        self.point, self.ishape, self.oshape = point, (_STATE,), (12,)
        super().__init__()

    def at(self, point):
        return _Dense(point)

    def forward(self, x, out=None):
        return _jacobian(self.point) @ x

    def adjoint(self, y, out=None):
        return _jacobian(self.point).conj().T @ y

    def normal(self, x, out=None):
        J = _jacobian(self.point)
        return J.conj().T @ (J @ x)

    def _as_callbacks(self):
        def plain(apply):
            def run(v):
                with torch.no_grad():
                    return apply(v)

            return run

        return linop.LinearOperator.from_callbacks(
            self.oshape, self.ishape, plain(self.forward), plain(self.adjoint), plain(self.normal)
        )


def _linearized():
    from bartorch.nlop.basic import Multiply
    from bartorch.nlop.step import flattened

    return flattened(Multiply(_IMAGE, _COILS).bundle)


_AT_A_POINT = {
    "ist": lambda: optim.ISTBlock(priors.ImplicitPrior(_Scale(0.9)), step=0.1),
    "fista": lambda: optim.FISTABlock(priors.ImplicitPrior(_Scale(0.9)), step=0.1),
    "admm": lambda: optim.ADMMBlock(priors.ImplicitPrior(_Scale(0.9)), rho=0.5, cg_maxiter=200),
    "pridu": lambda: optim.PRIDUBlock(priors.ImplicitPrior(_Scale(0.9)), step=0.5),
}


def _at_a_point(generator):
    def rand(*shape):
        real = torch.randn(*shape, generator=generator)
        return (real + 1j * torch.randn(*shape, generator=generator)).to(torch.complex64)

    return rand(_STATE) * 0.3 + 1.0, rand(12), rand(_STATE)


@pytest.mark.parametrize("name", sorted(_AT_A_POINT))
def test_a_block_over_a_linearization_is_differentiable_by_the_point(name):
    """Against the same block over the derivative written in torch."""
    point, y, seed = _at_a_point(torch.Generator().manual_seed(0))
    bundle = _linearized()
    answers = []
    for make in (bundle.at, _Dense):
        at = point.clone().requires_grad_(True)
        x = _run([_AT_A_POINT[name]()] * 3, y, make(at))
        (x.conj() * seed).sum().real.backward()
        answers.append((x.detach(), at.grad))
    (ours, our_grad), (theirs, their_grad) = answers
    assert torch.equal(ours, theirs)
    assert (our_grad - their_grad).abs().max() < 1e-6 * their_grad.abs().max()


def test_the_point_gradient_through_the_inner_solve_is_the_one_finite_differences_measure(
    monkeypatch,
):
    """With the inner solve exact; BART's own tolerance moves it by as much as that."""
    monkeypatch.setattr(optim.ADMMBlock, "_cg_eps", 0.0)
    point, y, seed = _at_a_point(torch.Generator().manual_seed(0))
    direction = _at_a_point(torch.Generator().manual_seed(1))[0]

    def loss(at):
        return (_run([_AT_A_POINT["admm"]()] * 3, y, _Dense(at)).conj() * seed).sum().real

    at = point.clone().requires_grad_(True)
    loss(at).backward()
    along = torch.real(torch.sum(at.grad.conj() * direction)).item()

    h = 3e-3
    with torch.no_grad():
        measured = (loss(point + h * direction) - loss(point - h * direction)).item() / (2 * h)
    assert abs(along - measured) <= 1e-3 * abs(measured)


def test_auxiliary_variables_are_refused_over_a_linearization():
    at = _at_a_point(torch.Generator().manual_seed(0))[0]
    block = optim.ADMMBlock(priors.TotalGeneralizedVariation((-1,), 0.01))
    with pytest.raises(TypeError, match="linearized at a point"):
        block.start(torch.zeros(12, dtype=torch.complex64), _linearized().at(at))


def test_conjugate_gradients_over_a_linearization_are_differentiable_by_the_point():
    """The solve ``IRGNM``'s second form hands to ``optim.CG``, against the torch derivative."""
    point, y, seed = _at_a_point(torch.Generator().manual_seed(0))
    solver = optim.CG(maxiter=200, tol=0.0, cclambda=0.5)
    answers = []
    for make in (_linearized().at, _Dense):
        at = point.clone().requires_grad_(True)
        x = solver(y, make(at))
        (x.conj() * seed).sum().real.backward()
        answers.append((x.detach(), at.grad))
    (ours, our_grad), (theirs, their_grad) = answers
    assert (ours - theirs).abs().max() < 1e-6 * theirs.abs().max()
    assert (our_grad - their_grad).abs().max() < 1e-5 * their_grad.abs().max()

    direction = _at_a_point(torch.Generator().manual_seed(1))[0]
    along = torch.real(torch.sum(their_grad.conj() * direction)).item()

    def loss(at):
        return (solver(y, _Dense(at)).conj() * seed).sum().real.item()

    h = 3e-3
    measured = (loss(point + h * direction) - loss(point - h * direction)) / (2 * h)
    assert abs(along - measured) <= 1e-3 * abs(measured)
