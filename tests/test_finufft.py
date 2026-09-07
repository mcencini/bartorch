"""The FINUFFT-backed NUFFT, against an explicit discrete Fourier sum and
against BART's own gridder.
"""

import numpy as np
import pytest
import torch

import bartorch.tools as bt
from bartorch import _finufft
from bartorch.ops import LinearOperator

requires_finufft = pytest.mark.skipif(not _finufft.available(), reason="finufft is not installed")


def _inner(a, b):
    return torch.vdot(a.flatten(), b.flatten()).real.item()


@requires_finufft
def test_matches_an_explicit_dft_on_a_radial_trajectory():
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    y = LinearOperator.finufft(traj, (1, n, n))(img)

    trj = traj.numpy().real
    im = img.numpy().reshape(n, n)
    kx, ky = trj[..., 0], trj[..., 1]
    x = np.arange(n) - n // 2
    phase = np.exp(
        -2j
        * np.pi
        * (
            kx[..., None, None] * x[None, None, None, :] / n
            + ky[..., None, None] * x[None, None, :, None] / n
        )
    )
    ref = (phase * im[None, None]).sum(axis=(-1, -2)) / n
    assert np.linalg.norm(y.numpy().reshape(ref.shape) - ref) / np.linalg.norm(ref) < 5e-3


@requires_finufft
def test_agrees_with_barts_own_gridder():
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    bart = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    fast = LinearOperator.finufft(traj, (1, n, n))
    assert fast.ishape == bart.ishape and fast.oshape == bart.oshape
    a, b = bart(img), fast(img)
    assert (a - b).norm().item() / a.norm().item() < 5e-3


@requires_finufft
def test_adjoint_identity_holds():
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    A = LinearOperator.finufft(traj, (1, n, n))
    x = torch.randn(*A.ishape, dtype=torch.complex64)
    y = torch.randn(*A.oshape, dtype=torch.complex64)
    assert _inner(A(x), y) == pytest.approx(_inner(x, A.adjoint(y)), rel=1e-3)


@requires_finufft
def test_it_carries_coils_through_one_plan():
    n, ncoils = 32, 4
    traj = bt.traj(x=n, y=16, r=True)
    A = LinearOperator.finufft(traj, (ncoils, n, n))
    assert A.oshape[0] == ncoils
    x = torch.randn(ncoils, n, n, dtype=torch.complex64)
    y = A(x)
    # Each coil must transform independently of the others.
    single = LinearOperator.finufft(traj, (1, n, n))
    for c in range(ncoils):
        torch.testing.assert_close(
            y[c].reshape(-1), single(x[c : c + 1]).reshape(-1), rtol=1e-4, atol=1e-4
        )


@requires_finufft
def test_bart_solves_against_a_finufft_operator():
    # The point of matching BART's convention: the operator goes into BART's
    # own conjugate gradients unchanged.
    n = 32
    traj = bt.traj(x=n, y=64, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    A = LinearOperator.finufft(traj, (1, n, n))
    y = A(img)
    x = A.lstsq(y, lambda_=1e-3, maxiter=30)
    assert x.shape == img.shape
    assert (x - img).norm().item() / img.norm().item() < 0.5
