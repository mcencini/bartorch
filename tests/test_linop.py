"""The linear operator surface: the classes, the algebra, and autograd.

What is checked here is what the split bought: that an operator is a class with
an interface, that one written in Python is the same kind of thing as one of
BART's and composes and solves with it, and that a backward pass is the adjoint
-- which for complex tensors is a claim about a conjugation and not only about
a transpose.
"""

import numpy as np
import pytest
import torch

import bartorch
from bartorch import linop, nlop, optim


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _inner(a, b):
    return torch.vdot(a.flatten(), b.flatten()).real.item()


# --- the interface ----------------------------------------------------------


def test_the_base_class_is_abstract():
    with pytest.raises(TypeError):
        linop.LinearOperator()


def test_a_subclass_says_how_it_is_applied():
    """Either BART builds it or Python applies it; half of the second is neither."""

    class ForwardOnly(linop.LinearOperator):
        def __init__(self):
            self.ishape = self.oshape = (4,)
            super().__init__()

        def forward(self, x, out=None):
            return x

    with pytest.raises(TypeError, match="forward and adjoint"):
        ForwardOnly()


def test_a_concrete_operator_is_backed_by_bart():
    F = linop.FFT((8, 16), axes=-1)
    assert isinstance(F, linop.LinearOperator)
    assert F._native and F._bart() is F
    assert F.ishape == F.oshape == (8, 16)
    assert "FFT" in repr(F)


def test_every_concrete_operator_says_how_it_is_built():
    for cls in (linop.FFT, linop.Diagonal, linop.Sampling, linop.MultiplySum, linop.NUFFT):
        assert "_create" in vars(cls), f"{cls.__name__} does not implement _create"


# --- an operator of one's own ----------------------------------------------


class Scale(linop.LinearOperator):
    """A linear operator written here rather than in BART."""

    def __init__(self, factor: complex, shape):
        self.factor = complex(factor)
        self.ishape = self.oshape = tuple(shape)

    def forward(self, x, out=None):
        return self.factor * x

    def adjoint(self, y, out=None):
        return self.factor.conjugate() * y


def test_an_operator_written_here_applies_and_has_the_adjoint_it_says():
    A = Scale(2 + 1j, (4, 8))
    x, y = _rand(4, 8), _rand(4, 8)
    torch.testing.assert_close(A(x), (2 + 1j) * x)
    assert _inner(A(x), y) == pytest.approx(_inner(x, A.adjoint(y)), rel=1e-4)


def test_an_operator_written_here_chains_with_one_of_barts():
    shape = (8, 16)
    S, F = Scale(3 - 2j, shape), linop.FFT(shape, axes=-1)
    A = F @ S
    x = _rand(*shape)
    torch.testing.assert_close(A(x), F(S(x)), rtol=1e-4, atol=1e-4)
    assert A._native
    assert not S._native


def test_an_operator_written_here_is_solved_by_barts_conjugate_gradients():
    A = Scale(2.0, (4, 4))
    y = _rand(4, 4)
    torch.testing.assert_close(optim.CG(maxiter=40)(y, A), y / 2, rtol=1e-3, atol=1e-4)


# --- the adjoint as an operator --------------------------------------------


def test_the_adjoint_of_an_operator_applies_without_building_a_second_one():
    F = linop.FFT((8, 16), axes=-1)
    y = _rand(8, 16)
    torch.testing.assert_close(F.H(y), F.adjoint(y))
    assert not hasattr(F.H, "_h"), "the adjoint built a BART handle it does not need"
    assert F.H.H is F


def test_the_adjoint_can_still_be_chained():
    shape = (8, 16)
    F = linop.FFT(shape, axes=-1)
    G = linop.FFT(shape, axes=-2)
    A = G @ F.H
    x = _rand(*shape)
    torch.testing.assert_close(A(x), G(F.adjoint(x)), rtol=1e-4, atol=1e-4)


# --- autograd ---------------------------------------------------------------


def test_the_backward_pass_is_the_adjoint_and_not_the_transpose():
    """The near miss differs by a conjugation, which a real test would not see."""
    shape = (4, 8)
    d, w = _rand(*shape), _rand(*shape)
    A = linop.Diagonal(d, shape)

    x = _rand(*shape).requires_grad_(True)
    ((A(x).conj() * w).sum().real).backward()

    # What torch itself makes of the same multiplication.
    ref = x.detach().clone().requires_grad_(True)
    (((d * ref).conj() * w).sum().real).backward()
    torch.testing.assert_close(x.grad, ref.grad, rtol=1e-4, atol=1e-5)

    # And what it would have been with the transpose instead of the adjoint,
    # which is what this test exists to tell apart.
    wrong = x.detach().clone().requires_grad_(True)
    (((d.conj() * wrong).conj() * w).sum().real).backward()
    assert not torch.allclose(x.grad, wrong.grad, rtol=1e-3, atol=1e-4)


def test_a_gradient_flows_through_a_chain_of_operators():
    shape = (8, 16)
    A = linop.FFT(shape, axes=-1) @ linop.Diagonal(_rand(*shape), shape)
    x = _rand(*shape).requires_grad_(True)
    y = A(x)
    y.abs().square().sum().backward()
    assert x.grad is not None and x.grad.shape == shape


def test_a_real_input_gets_a_real_gradient():
    shape = (4, 8)
    A = linop.Diagonal(_rand(*shape), shape)
    x = torch.randn(*shape, requires_grad=True)
    A(x).abs().square().sum().backward()
    assert x.grad.dtype == torch.float32
    assert x.grad.shape == shape


def test_the_adjoint_differentiates_too():
    shape = (4, 8)
    A = linop.Diagonal(_rand(*shape), shape)
    y = _rand(*shape).requires_grad_(True)
    A.A_adjoint(y).abs().square().sum().backward()
    assert y.grad is not None


def test_applying_an_operator_to_a_plain_tensor_records_nothing():
    A = linop.FFT((4, 8), axes=-1)
    assert A(_rand(4, 8)).grad_fn is None


def test_writing_into_a_buffer_and_tracking_a_gradient_do_not_mix():
    shape = (4, 8)
    A = linop.FFT(shape, axes=-1)
    out = torch.empty(shape, dtype=torch.complex64)
    with pytest.raises(ValueError, match="autograd"):
        A(_rand(*shape).requires_grad_(True), out=out)


def test_a_torch_model_can_hold_a_bart_operator():
    """What the recording is for: a step of gradient descent through the encoding."""
    shape = (2, 8, 8)
    maps = _rand(*shape)
    A = linop.MultiplySum(maps, (1, 8, 8), shape)
    truth = _rand(1, 8, 8)
    data = A(truth)

    x = torch.zeros(1, 8, 8, dtype=torch.complex64, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.3)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = (A(x) - data).abs().square().sum()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first / 10


# --- the nonlinear side -----------------------------------------------------


def test_a_nonlinear_operator_linearises_into_a_linear_one():
    t = torch.linspace(0, 1, 16, dtype=torch.complex64)

    def model(p):
        return p[0] * torch.exp(-t * p[1])

    F = nlop.FromTorch(model, ishape=(2,), oshape=(16,))
    x = torch.tensor([2.0, 1.5], dtype=torch.complex64)
    D = F.linearize(x)
    assert isinstance(D, linop.LinearOperator)
    dx, dy = _rand(2), _rand(16)
    assert _inner(D(dx), dy) == pytest.approx(_inner(dx, D.adjoint(dy)), rel=1e-3)


def test_a_nonlinear_operators_backward_pass_is_its_adjoint_derivative():
    t = torch.linspace(0.1, 1, 16, dtype=torch.complex64)

    def model(p):
        return p[0] * torch.exp(-t * p[1])

    F = nlop.FromTorch(model, ishape=(2,), oshape=(16,))
    w = _rand(16)

    x = torch.tensor([2.0, 1.5], dtype=torch.complex64).requires_grad_(True)
    (F(x).conj() * w).sum().real.backward()

    ref = x.detach().clone().requires_grad_(True)
    (model(ref).conj() * w).sum().real.backward()
    torch.testing.assert_close(x.grad, ref.grad, rtol=1e-3, atol=1e-4)


# --- deepinv ----------------------------------------------------------------


def test_an_operator_answers_to_deepinvs_names_without_deepinv_installed():
    shape = (2, 8, 8)
    A = linop.MultiplySum(_rand(*shape), (1, 8, 8), shape)
    x = _rand(1, 8, 8)
    torch.testing.assert_close(A.A(x), A(x))
    torch.testing.assert_close(A.A_adjoint(A(x)), A.adjoint(A(x)))
    assert A.A_dagger(A(x)).shape == (1, 8, 8)


def test_an_operator_becomes_a_linear_physics():
    deepinv = pytest.importorskip("deepinv")
    shape = (2, 8, 8)
    maps = _rand(*shape)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    A = linop.MultiplySum(maps, (1, 8, 8), shape)

    physics = bartorch.to_deepinv(A)
    assert isinstance(physics, deepinv.physics.LinearPhysics)

    x = _rand(1, 8, 8)
    y = physics(x)
    torch.testing.assert_close(y, A(x), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(physics.A_adjoint(y), A.adjoint(y), rtol=1e-4, atol=1e-5)
    # A^H A is the identity for orthonormal maps, so the solve returns x.
    torch.testing.assert_close(physics.A_dagger(y), x, rtol=1e-3, atol=1e-4)


def test_the_physics_walks_deepinvs_batch_axis():
    """An operator's shape is fixed; one axis more than it expects is a batch."""
    pytest.importorskip("deepinv")
    shape = (2, 8, 8)
    maps = _rand(*shape)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    physics = bartorch.to_deepinv(linop.MultiplySum(maps, (1, 8, 8), shape))

    batch = torch.stack([_rand(1, 8, 8) for _ in range(3)])
    y = physics.A(batch)
    assert y.shape == (3, *shape)
    torch.testing.assert_close(physics.A_dagger(y), batch, rtol=1e-3, atol=1e-4)
    for i in range(3):
        torch.testing.assert_close(y[i], physics.op(batch[i]), rtol=1e-4, atol=1e-5)


def test_a_gradient_flows_through_the_physics():
    pytest.importorskip("deepinv")
    shape = (2, 8, 8)
    physics = bartorch.to_deepinv(linop.MultiplySum(_rand(*shape), (1, 8, 8), shape))
    x = torch.stack([_rand(1, 8, 8) for _ in range(2)]).requires_grad_(True)
    physics.A(x).abs().square().sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_the_physics_is_not_what_an_operator_inherits_from():
    """deepinv is an adapter, not a base class, so it is never in the way."""
    for cls in linop.FFT.__mro__:
        assert not cls.__module__.startswith("deepinv"), (
            f"{cls} makes deepinv a dependency of every operator"
        )


# --- what BART's own operators still do -------------------------------------


def test_the_fft_operator_still_matches_numpy():
    x = _rand(8, 16)
    F = linop.FFT((8, 16), axes=-1)
    ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x.numpy(), axes=-1), axis=-1), axes=-1)
    np.testing.assert_allclose(F(x).numpy(), ref / np.sqrt(16), rtol=1e-4, atol=1e-4)
