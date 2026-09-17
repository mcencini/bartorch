"""The adapters a network needs: a real-valued denoiser over complex images, and a stack.

:class:`~bartorch.learning.Denoiser` is held against the arithmetic written
out in torch, which is what says the planes it builds carry what it claims.
:class:`~bartorch.learning.Unrolled` is held against the solver it is the loop
of: frozen, the stack is BART's iteration to the bit, and neither
``checkpoint`` nor the choice of a shared block changes what a step computes.
"""

import pytest
import torch
from torch import nn

from bartorch import learning, linop, optim, priors

SHAPE = (1, 8, 8)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


@pytest.fixture
def problem():
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=(-1, -2))
    return A, A(_rand(*SHAPE))


class _Seen(nn.Module):
    """A network that answers with its input and remembers what it was given."""

    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.seen = []

    def forward(self, x, sigma=None):
        self.seen.append((tuple(x.shape), sigma))
        return self.gain * x


# --- the pair of real channels ------------------------------------------------------


def test_the_pair_is_a_leading_axis_and_comes_back():
    x = _rand(3, 4, 5)
    made = learning.as_real(x)
    assert (2, 3, 4, 5) == tuple(made.shape) and not made.is_complex()
    assert torch.equal(made[0], x.real) and torch.equal(made[1], x.imag)
    assert torch.equal(learning.as_complex(made), x)


def test_a_real_image_is_carried_with_nothing_imaginary():
    x = torch.randn(4, 4)
    assert torch.equal(learning.as_complex(learning.as_real(x)).real, x)
    assert 0.0 == float(learning.as_real(x)[1].abs().sum())


def test_as_complex_refuses_what_as_real_did_not_make():
    with pytest.raises(ValueError, match="leading axis"):
        learning.as_complex(torch.randn(3, 4))
    with pytest.raises(TypeError, match="complex already"):
        learning.as_complex(_rand(2, 4))


# --- the denoiser adapter -----------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "channels", "planes"),
    [("channels", 2, 2), ("separate", 1, 1), ("separate", 3, 3), ("magnitude", 1, 1)],
)
def test_the_network_is_given_planes_of_its_own_rank_and_channel_count(parts, channels, planes):
    net = _Seen()
    denoiser = learning.Denoiser(net, channels=channels, parts=parts, normalize=False)
    x = _rand(3, 5, 8, 8)

    out = denoiser(x)

    images = 3 * 5 * (2 if "separate" == parts else 1)
    assert [((images, planes, 8, 8), None)] == net.seen
    assert tuple(out.shape) == tuple(x.shape) and out.is_complex()


def test_the_planes_are_the_parts_the_mode_names():
    """Against the parts written out, which is what says the layout is what it claims."""

    class _Keep(nn.Module):
        def forward(self, v):
            self.planes = v.clone()
            return v

    x = _rand(2, 8, 8)

    together = _Keep()
    learning.Denoiser(together, channels=2, normalize=False)(x)
    assert torch.equal(together.planes[:, 0], x.real)
    assert torch.equal(together.planes[:, 1], x.imag)

    apart = _Keep()
    learning.Denoiser(apart, channels=1, parts="separate", normalize=False)(x)
    assert torch.equal(apart.planes[:2, 0], x.real)
    assert torch.equal(apart.planes[2:, 0], x.imag)

    modulus = _Keep()
    learning.Denoiser(modulus, channels=1, parts="magnitude", normalize=False)(x)
    assert torch.equal(modulus.planes[:, 0], x.abs())


@pytest.mark.parametrize("parts", ["channels", "separate", "magnitude"])
def test_the_identity_network_leaves_the_image_alone(parts):
    channels = 2 if "channels" == parts else 1
    denoiser = learning.Denoiser(_Seen(), channels=channels, parts=parts)
    x = 17.0 * _rand(2, 3, 8, 8)
    assert torch.allclose(denoiser(x), x, atol=1e-5)


def test_the_phase_survives_a_magnitude_network():
    class _Blur(nn.Module):
        def forward(self, v):
            return 0.5 * v

    x = _rand(2, 8, 8)
    out = learning.Denoiser(_Blur(), channels=1, parts="magnitude")(x)
    assert torch.allclose(out.angle(), x.angle(), atol=1e-4)
    assert torch.allclose(out.abs(), 0.5 * x.abs(), atol=1e-5)


def test_an_rgb_network_is_given_three_copies_and_answers_for_one():
    class _Weighted(nn.Module):
        def forward(self, v):
            assert 3 == v.shape[1]
            return v * torch.tensor([0.0, 3.0, 0.0]).reshape(1, 3, 1, 1)

    x = _rand(2, 8, 8)
    out = learning.Denoiser(_Weighted(), channels=3, parts="separate")(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_each_item_of_the_leading_axis_is_scaled_by_its_own_peak():
    class _Ones(nn.Module):
        def forward(self, v):
            return torch.ones_like(v)

    x = torch.stack([_rand(4, 4), 100.0 * _rand(4, 4)])

    # The network answers one everywhere, so the modulus that comes back is
    # the scale each item was divided by, which is its own peak and not the
    # batch's.
    out = learning.Denoiser(_Ones(), channels=1, parts="magnitude")(x)

    assert torch.allclose(out[0].abs(), x[0].abs().amax().expand(4, 4), atol=1e-4)
    assert torch.allclose(out[1].abs(), x[1].abs().amax().expand(4, 4), atol=1e-2)


def test_without_normalization_the_values_reach_the_network_as_they_are():
    class _Keep(nn.Module):
        def forward(self, v):
            self.peak = float(v.abs().amax())
            return v

    x = 100.0 * _rand(1, 4, 4)

    scaled, plain = _Keep(), _Keep()
    learning.Denoiser(scaled, channels=2)(x)
    learning.Denoiser(plain, channels=2, normalize=False)(x)

    assert scaled.peak <= 1.0 + 1e-6
    assert plain.peak == pytest.approx(float(torch.stack([x.real, x.imag]).abs().amax()))


def test_a_volume_network_takes_three_spatial_axes():
    net = _Seen()
    learning.Denoiser(net, spatial=3, channels=2)(_rand(2, 3, 4, 5, 6))
    assert [((6, 2, 4, 5, 6), None)] == net.seen


def test_one_image_with_no_batch_is_a_batch_of_one():
    net = _Seen()
    learning.Denoiser(net, channels=2)(_rand(8, 8))
    assert [((1, 2, 8, 8), None)] == net.seen


def test_a_real_image_comes_back_real():
    out = learning.Denoiser(_Seen(), channels=2)(torch.randn(2, 8, 8))
    assert not out.is_complex() and torch.float32 == out.dtype


def test_the_noise_level_is_passed_through_only_when_it_is_given():
    net = _Seen()
    denoiser = learning.Denoiser(net, channels=2)
    denoiser(_rand(1, 8, 8))
    denoiser(_rand(1, 8, 8), 0.05)
    assert [None, 0.05] == [sigma for _, sigma in net.seen]


def test_the_gradient_reaches_the_image_and_the_networks_weights():
    net = _Seen()
    x = _rand(1, 8, 8).requires_grad_()
    learning.Denoiser(net, channels=2)(x).abs().square().sum().backward()
    assert x.grad is not None and net.gain.grad is not None and 0 != net.gain.grad


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 3, "parts": "channels"}, "two of them"),
        ({"channels": 2, "parts": "separate"}, "neither"),
        ({"channels": 1, "parts": "wavelet"}, "parts is one of"),
        ({"channels": 2, "spatial": 4}, "2 or 3"),
    ],
)
def test_a_layout_the_adapter_cannot_make_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        learning.Denoiser(_Seen(), **kwargs)


def test_a_network_that_answers_the_wrong_shape_is_named():
    class _Crop(nn.Module):
        def forward(self, v):
            return v[..., :4, :4]

    with pytest.raises(ValueError, match="a denoiser returns"):
        learning.Denoiser(_Crop(), channels=2)(_rand(1, 8, 8))


def test_a_denoiser_stands_where_a_regularizer_does(problem):
    A, y = problem
    net = _Seen(0.9)
    term = priors.ImplicitPrior(learning.Denoiser(net, channels=2))
    out = optim.fista(y, A, term, maxiter=3)
    assert tuple(out.shape) == SHAPE and 0 != len(net.seen)


# --- the stack ----------------------------------------------------------------------


def _block():
    return optim.ADMMBlock(priors.ImplicitPrior(_Seen(0.9)), rho=0.5, cg_maxiter=20)


def test_a_frozen_stack_is_the_solver_to_the_bit(problem):
    A, y = problem
    denoiser = _Seen(0.9)
    term = priors.ImplicitPrior(denoiser)
    stack = learning.Unrolled(optim.ADMMBlock(term, rho=0.5, cg_maxiter=20), iterations=5)
    solver = optim.ADMM(term, maxiter=5, rho=0.5, cg_maxiter=20)
    assert torch.equal(stack(y, A), solver(y, A))


def test_one_block_is_shared_by_every_iteration(problem):
    A, y = problem
    block = _block()
    stack = learning.Unrolled(block, iterations=4)
    assert all(stack.block(k) is block for k in range(4))
    assert len(list(stack.steps(y, A))) == 4


def test_a_block_per_iteration_is_taken_in_turn(problem):
    A, y = problem
    blocks = [_block() for _ in range(3)]
    stack = learning.Unrolled(blocks)
    assert 3 == stack.iterations
    assert [stack.block(k) for k in range(3)] == blocks
    assert {id(p) for p in stack.parameters()} >= {id(p) for b in blocks for p in b.parameters()}


def test_the_steps_end_where_the_stack_does(problem):
    A, y = problem
    stack = learning.Unrolled(_block(), iterations=4)
    assert torch.equal(list(stack.steps(y, A))[-1], stack(y, A))


def test_checkpointing_changes_neither_the_image_nor_the_gradient(problem):
    A, y = problem
    grads, images = [], []
    for checkpoint in (False, True):
        torch.manual_seed(3)
        block = _block()
        block.rho.requires_grad_()
        image = learning.Unrolled(block, iterations=4, checkpoint=checkpoint)(y, A)
        image.abs().square().sum().backward()
        images.append(image.detach())
        grads.append((float(block.rho.grad), float(block.terms[0].denoiser.gain.grad)))

    assert torch.equal(*images)
    assert grads[0] == pytest.approx(grads[1], rel=1e-5)


def test_a_detached_stack_is_the_same_image_and_the_last_steps_gradient(problem):
    A, y = problem
    made = []
    for detach in (False, True):
        block = _block()
        block.rho.requires_grad_()
        image = learning.Unrolled(block, iterations=4, detach=detach)(y, A)
        image.abs().square().sum().backward()
        made.append((image.detach(), float(block.rho.grad)))

    assert torch.equal(made[0][0], made[1][0])
    assert made[0][1] != made[1][1], "a detached stack would carry the whole stack's gradient"


def test_a_stack_needs_to_know_how_many_iterations():
    with pytest.raises(ValueError, match="how many"):
        learning.Unrolled(_block())
    with pytest.raises(ValueError, match="not 7"):
        learning.Unrolled([_block(), _block()], 7)
    with pytest.raises(ValueError, match="computes nothing"):
        learning.Unrolled([])
