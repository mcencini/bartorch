"""A block at its fixed point: the answer is one, and the gradient is the point's."""

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
    def __init__(self, weight: float = 0.8):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, x, sigma=None):
        return self.weight * x


def test_with_nothing_to_learn_it_is_the_block_run_that_many_times(problem):
    A, y = problem
    block = optim.ISTBlock(priors.L1(0.02), step=0.7)
    deq = optim.FixedPoint(block, max_iter=300, tol=1e-6)
    made = deq(y, A)
    assert 1 < deq.iterations < 300

    state = block.start(y, A)
    for _ in range(deq.iterations):
        state = block(state, A)
    assert torch.equal(made, block.output(state, A))


def test_the_answer_is_a_fixed_point(problem):
    A, y = problem
    block = optim.PRIDUBlock(priors.L1(0.02), step=0.95)
    deq = optim.FixedPoint(block, max_iter=500, tol=1e-7)
    x = deq(y, A)

    state = block.start(y, A, x)
    for _ in range(3):
        state = block(state, A)
    assert (state.x - x).abs().max() < 1e-2 * x.abs().max()


@pytest.mark.parametrize(
    "make",
    [
        lambda p: optim.ISTBlock(p, step=0.7),
        lambda p: optim.PRIDUBlock(p, step=0.5),
        lambda p: optim.ADMMBlock(p, rho=0.5, cg_maxiter=20),
    ],
    ids=["ist", "pridu", "admm"],
)
def test_the_gradient_is_the_one_finite_differences_measure(problem, make):
    A, y = problem

    def loss(weight):
        denoiser = _Scale()
        if isinstance(weight, torch.Tensor):
            denoiser.weight = weight
        else:
            denoiser.weight.data.fill_(weight)
            denoiser.weight.requires_grad_(False)
        deq = optim.FixedPoint(make(priors.ImplicitPrior(denoiser)), max_iter=400, tol=1e-9)
        return deq(y, A).abs().square().sum()

    weight = nn.Parameter(torch.tensor(0.8))
    loss(weight).backward()

    h = 1e-3
    measured = (loss(0.8 + h) - loss(0.8 - h)).item() / (2 * h)
    assert abs(weight.grad.item() - measured) <= 1e-2 * abs(measured)


def test_the_data_carries_a_gradient(problem):
    A, y = problem
    data = y.clone().requires_grad_()
    deq = optim.FixedPoint(optim.ISTBlock(priors.ImplicitPrior(_Scale()), step=0.7))
    deq(data, A).abs().square().sum().backward()
    assert torch.isfinite(data.grad).all() and 0 < data.grad.abs().sum()


def test_the_denoisers_weights_are_the_models(problem):
    denoiser = _Scale()
    deq = optim.FixedPoint(optim.ISTBlock(priors.ImplicitPrior(denoiser)))
    assert any(p is denoiser.weight for p in deq.parameters())


def test_a_batch_is_the_same_answers_side_by_side(problem):
    A, _ = problem
    data = torch.stack([A(_rand(*SHAPE)) for _ in range(2)])
    deq = optim.FixedPoint(optim.ISTBlock(priors.L1(0.02), step=0.7), max_iter=40, tol=0.0)
    batched = deq(data, A)
    for i in range(2):
        assert torch.equal(batched[i], deq(data[i], A))


def test_a_step_that_changes_has_no_fixed_point():
    with pytest.raises(TypeError, match="momentum"):
        optim.FixedPoint(optim.FISTABlock(priors.L1(0.02)))
    with pytest.raises(ValueError, match="rho that moves"):
        optim.FixedPoint(optim.ADMMBlock(priors.L1(0.02), dynamic_rho=True))
    with pytest.raises(ValueError, match="steps that move"):
        optim.FixedPoint(optim.PRIDUBlock(priors.L1(0.02), adaptive_step=True))
