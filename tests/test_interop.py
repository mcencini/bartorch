"""The deepinv adapter: an operator as a physics, and deepinv only where it is asked for."""

import subprocess
import sys

import pytest
import torch

from bartorch import interop, linop

deepinv = pytest.importorskip("deepinv")


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def test_importing_bartorch_and_solving_does_not_import_deepinv():
    script = (
        "import sys, torch\n"
        "from bartorch import linop, optim, priors\n"
        "A = linop.FFT((1, 8, 8), axes=(-1, -2))\n"
        "y = A(torch.randn(1, 8, 8, dtype=torch.complex64))\n"
        "optim.fista(y, A, priors.L1(0.01), maxiter=2)\n"
        "assert 'deepinv' not in sys.modules, 'deepinv was imported'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_an_operator_becomes_a_linear_physics():
    shape = (2, 8, 8)
    maps = _rand(*shape)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    A = linop.MultiplySum(maps, (1, 8, 8), shape)

    physics = interop.to_deepinv(A)
    assert isinstance(physics, deepinv.physics.LinearPhysics)

    x = _rand(1, 8, 8)
    y = physics(x)
    torch.testing.assert_close(y, A(x), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(physics.A_adjoint(y), A.adjoint(y), rtol=1e-4, atol=1e-5)
    # A^H A is the identity for orthonormal maps, so the solve returns x.
    torch.testing.assert_close(physics.A_dagger(y), x, rtol=1e-3, atol=1e-4)


def test_the_physics_walks_deepinvs_batch_axis():
    """An operator's shape is fixed; one axis more than it expects is a batch."""
    shape = (2, 8, 8)
    maps = _rand(*shape)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    physics = interop.to_deepinv(linop.MultiplySum(maps, (1, 8, 8), shape))

    batch = torch.stack([_rand(1, 8, 8) for _ in range(3)])
    y = physics.A(batch)
    assert y.shape == (3, *shape)
    torch.testing.assert_close(physics.A_dagger(y), batch, rtol=1e-3, atol=1e-4)
    for i in range(3):
        torch.testing.assert_close(y[i], physics.op(batch[i]), rtol=1e-4, atol=1e-5)


def test_a_gradient_flows_through_the_physics():
    shape = (2, 8, 8)
    physics = interop.to_deepinv(linop.MultiplySum(_rand(*shape), (1, 8, 8), shape))
    x = torch.stack([_rand(1, 8, 8) for _ in range(2)]).requires_grad_(True)
    physics.A(x).abs().square().sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_the_physics_is_not_what_an_operator_inherits_from():
    """deepinv is an adapter, not a base class, so it is never in the way."""
    assert not hasattr(linop.FFT((8, 8), axes=-1), "A_adjoint")
    for cls in linop.FFT.__mro__:
        assert not cls.__module__.startswith("deepinv"), (
            f"{cls} makes deepinv a dependency of every operator"
        )
