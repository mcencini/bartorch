"""Operators: BART's own, ones defined in Python, and BART's solvers on both,
checked against numpy and closed forms.
"""

import numpy as np
import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import linop, nlop, optim
from bartorch.linop import basic


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _inner(a, b):
    return torch.vdot(a.flatten(), b.flatten()).real.item()


def test_bart_fft_operator_matches_numpy():
    x = _rand(8, 16)
    F = linop.FFT((8, 16), axes=-1)
    y = F(x)
    ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x.numpy(), axes=-1), axis=-1), axes=-1)
    np.testing.assert_allclose(y.numpy(), ref / np.sqrt(16), rtol=1e-4, atol=1e-4)


def test_adjoint_identity_holds_for_the_fft_operator():
    F = linop.FFT((8, 16), axes=(-1, -2))
    x, y = _rand(8, 16), _rand(8, 16)
    assert _inner(F(x), y) == pytest.approx(_inner(x, F.adjoint(y)), rel=1e-4)


def test_python_operator_runs_inside_bart_and_chains_with_a_bart_operator():
    shape = (8, 16)
    w = _rand(*shape)
    W = linop.Callback(shape, shape, lambda x: w * x, lambda y: w.conj() * y)
    F = linop.FFT(shape, axes=-1)
    A = F @ W
    x = _rand(*shape)
    torch.testing.assert_close(A(x), F(w * x), rtol=1e-4, atol=1e-4)
    y = _rand(*shape)
    assert _inner(A(x), y) == pytest.approx(_inner(x, A.adjoint(y)), rel=1e-4)


def test_bart_uses_the_normal_callback_when_given_one():
    shape = (4, 4)
    calls = []

    def normal(x):
        calls.append(1)
        return 2 * x

    Op = linop.Callback(shape, shape, lambda x: x, lambda y: y, normal=normal)
    x = _rand(*shape)
    torch.testing.assert_close(Op.normal(x), 2 * x)
    assert calls


def test_least_squares_recovers_the_image_from_coil_data():
    n, ncoils = 16, 4
    ksp = bt.phantom([n, n], kspace=True, coils=ncoils)
    maps = bt.ecalib(ksp, calib_size=12, maps=1).reshape(ncoils, n, n)
    img = bt.phantom([n, n]).reshape(1, n, n)
    S = linop.MultiplySum(maps, (1, n, n), (ncoils, n, n))
    F = linop.FFT((ncoils, n, n), axes=(-1, -2))
    A = F @ S
    y = A(img)
    x = optim.CG(maxiter=50, tol=1e-8)(y, A)
    assert ((x - img).norm() / img.norm()).item() < 1e-3


def test_sampling_operator_zeroes_unsampled_lines():
    shape = (4, 8, 8)
    pattern = torch.zeros(1, 8, 8, dtype=torch.complex64)
    pattern[..., ::2, :] = 1
    P = basic.Sampling(pattern, shape)
    x = _rand(*shape)
    torch.testing.assert_close(P(x), x * pattern)


def test_nufft_operator_agrees_with_the_nufft_tool():
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    A = linop.NUFFT(traj, (1, n, n))
    y = A(img)
    ref = bartorch.nufft(img, traj)
    torch.testing.assert_close(y.reshape(ref.shape), ref, rtol=1e-3, atol=1e-3)
    z = _rand(*A.oshape)
    assert _inner(A(img), z) == pytest.approx(_inner(img, A.adjoint(z)), rel=1e-3)


def test_toeplitz_normal_operator_matches_adjoint_after_forward():
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    A = linop.NUFFT(traj, (1, n, n), toeplitz=True)
    torch.testing.assert_close(A.normal(img), A.adjoint(A(img)), rtol=2e-2, atol=2e-2)


def test_torch_function_becomes_a_bart_nonlinear_operator_with_a_correct_adjoint():
    shape = (6,)

    def fn(x):
        return x * x.abs() + x.conj() * 0.5

    F = nlop.FromTorch(fn, shape, shape)
    x = _rand(*shape)
    torch.testing.assert_close(F(x), fn(x))
    dx, dy = _rand(*shape), _rand(*shape)
    eps = 1e-3
    fd = (fn(x + eps * dx) - fn(x - eps * dx)) / (2 * eps)
    torch.testing.assert_close(F.derivative(dx), fd, rtol=1e-2, atol=1e-2)
    assert _inner(F.derivative(dx), dy) == pytest.approx(_inner(dx, F.adjoint(dy)), rel=1e-3)


def test_gauss_newton_fits_a_mono_exponential_decay():
    # A signal model as torchsim produces one: parameters (amplitude, rate)
    # per voxel, echoes along the last axis.  The closed-form data is fitted
    # back to the parameters by BART's IRGNM through the torch derivative.
    nvox, nechoes = 5, 12
    t = torch.linspace(0, 2.0, nechoes)
    truth = torch.stack([torch.full((nvox,), 1.5), torch.linspace(0.6, 1.4, nvox)]).to(
        torch.complex64
    )

    def model(p):
        return p[0][:, None] * torch.exp(-p[1][:, None] * t[None, :])

    F = nlop.FromTorch(model, (2, nvox), (nvox, nechoes))
    y = model(truth)
    x0 = torch.ones(2, nvox, dtype=torch.complex64)
    gauss_newton = optim.IRGNM(iterations=10, alpha=1.0, alpha_min=1e-6, redu=3.0, cg_maxiter=50)
    x = gauss_newton(y, F, x0)
    torch.testing.assert_close(x, truth, rtol=1e-2, atol=1e-2)


def test_model_based_reconstruction_chains_a_torch_model_with_a_bart_encoding():
    # Parameter maps -> signal images (torch) -> coil k-space (BART), solved
    # jointly by Gauss-Newton: the model-based reconstruction pattern.
    n, nechoes = 8, 6
    t = torch.linspace(0, 1.0, nechoes)
    img = bt.phantom([n, n]).reshape(n, n).real
    truth = torch.stack([img, 0.5 + 0.5 * img]).to(torch.complex64)

    def model(p):
        return p[0][None] * torch.exp(-p[1][None] * t[:, None, None])

    F = linop.FFT((nechoes, n, n), axes=(-1, -2))
    M = nlop.FromTorch(model, (2, n, n), (nechoes, n, n))
    A = F @ M
    assert A.ishape == (2, n, n) and A.oshape == (nechoes, n, n)
    y = A(truth)
    x0 = torch.stack([torch.ones(n, n), 0.5 * torch.ones(n, n)]).to(torch.complex64)
    gauss_newton = optim.IRGNM(iterations=12, alpha=1.0, alpha_min=1e-6, redu=3.0, cg_maxiter=60)
    x = gauss_newton(y, A, x0)
    mask = img > 0.1
    err = (x[0][mask] - truth[0][mask]).abs().max().item()
    assert err < 5e-2


def test_a_three_dimensional_nufft_takes_three_spatial_axes_from_the_trajectory():
    """(coils, y, x) and (z, y, x) are the same shape; the trajectory decides.

    A BART trajectory always carries three components and leaves kz at zero
    for a two-dimensional transform, so that is what says how many of an
    image's trailing axes are spatial and how many are coils.
    """
    n, coils, spokes = 16, 2, 40
    torch.manual_seed(0)
    volumetric = ((torch.rand(spokes, n, 3) - 0.5) * n).to(torch.complex64)
    planar = volumetric.clone()
    planar[..., 2] = 0

    # A volume: the last three axes are spatial and the coils sit in front.
    A = linop.NUFFT(volumetric, (coils, n, n, n), toeplitz=False)
    assert A.oshape == (coils, spokes, n)

    # The same shape read two-dimensionally makes its first axis a batch
    # rather than a third spatial axis, so kz staying at zero is what says so.
    B = linop.NUFFT(planar, (n, n, n), toeplitz=False)
    assert B.oshape == (n, spokes, n)
