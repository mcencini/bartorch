"""Reconstruction tools end to end, against references computed outside BART."""

import numpy as np
import pytest
import torch

import bartorch.tools as bt


def _fully_sampled_coil_kspace(n=32, coils=4):
    ksp = bt.phantom([n, n], kspace=True, coils=coils)
    assert ksp.shape[-2:] == (n, n)
    return ksp


def test_analytical_phantom_kspace_transforms_back_to_the_sampled_phantom():
    # phantom -k is the closed-form Fourier transform of the ellipses, scaled
    # for BART's non-unitary inverse, so it agrees with the sampled image
    # only up to discretisation.
    n = 64
    img = bt.phantom([n, n]).squeeze()
    ksp = bt.phantom([n, n], kspace=True).squeeze()
    back = bt.ifft(ksp, axes=(-1, -2))
    a, b = back.abs().flatten().numpy(), img.abs().flatten().numpy()
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_ecalib_maps_are_normalised_where_the_object_is():
    ksp = _fully_sampled_coil_kspace()
    maps = bt.ecalib(ksp, calib_size=16, maps=1)
    coil_axis = -3 if maps.dim() == 3 else -4
    norm = (maps.abs() ** 2).sum(coil_axis).sqrt().squeeze()
    centre = norm[norm.shape[0] // 2, norm.shape[1] // 2]
    assert centre == pytest.approx(1.0, abs=1e-2)


def test_pics_on_fully_sampled_data_matches_the_direct_inverse():
    ksp = _fully_sampled_coil_kspace()
    maps = bt.ecalib(ksp, calib_size=16, maps=1)
    reco = bt.pics(ksp, maps, l2=0.001, maxiter=30, l=2).squeeze()
    coil_images = bt.ifft(ksp, axes=(-1, -2), unitary=True).squeeze()
    direct = (coil_images * maps.squeeze().conj()).sum(0)
    assert reco.shape == direct.shape
    scale = (reco.abs().max() / direct.abs().max()).item()
    err = (reco.abs() / scale - direct.abs()).norm() / direct.abs().norm()
    assert err < 0.05


def test_nufft_matches_an_explicit_dft_on_a_radial_trajectory():
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ksp = bt.nufft(traj, img).squeeze()
    trj = traj.numpy().real  # (spokes, samples, 3): kx, ky, kz in grid units
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
    assert ksp.shape == ref.shape
    diff = ksp.numpy() - ref
    assert np.linalg.norm(diff) / np.linalg.norm(ref) < 5e-3
    assert np.abs(diff).max() / np.abs(ref).max() < 2e-2


def test_whitening_makes_noise_covariance_the_identity():
    ncoils, nsamples = 4, 4096
    rng = np.random.default_rng(0)
    mixing = rng.standard_normal((ncoils, ncoils)) + 1j * rng.standard_normal((ncoils, ncoils))
    noise = rng.standard_normal((nsamples, ncoils)) + 1j * rng.standard_normal((nsamples, ncoils))
    noise = (noise @ mixing.T).astype(np.complex64)
    noise_t = torch.from_numpy(noise).reshape(1, ncoils, 1, 1, nsamples).movedim(-1, -2)
    white = bt.whiten(noise_t, noise_t)
    w = white.reshape(ncoils, nsamples).numpy()
    cov = (w @ w.conj().T) / nsamples
    np.testing.assert_allclose(cov, np.eye(ncoils), atol=0.1)
