"""The denoisers against properties of their objectives, not against BART."""

from __future__ import annotations

import pytest
import torch

from bartorch._dispatch import BartError
from bartorch.prox import nlmeans, rof, tgv

N = 16


def _noise(shape, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    return sigma * torch.randn(shape, dtype=torch.complex64, generator=g)


def _square(sigma=0.1, seed=0):
    clean = torch.zeros(N, N, dtype=torch.complex64)
    clean[N // 4 : 3 * N // 4, N // 4 : 3 * N // 4] = 1
    return clean, clean + _noise((N, N), sigma, seed)


def _tv(a, axes=(0, 1)):
    """Anisotropic periodic TV, an outside measure of roughness."""
    return sum((a - a.roll(1, ax)).abs().sum().item() for ax in axes)


def _err(a, b):
    return (a - b).norm().item()


TV_DENOISERS = [
    pytest.param(lambda y, lam, axes: rof(y, lam, axes), id="rof"),
    pytest.param(lambda y, lam, axes: tgv(y, lam, axes), id="tgv"),
]


@pytest.mark.parametrize("denoise", TV_DENOISERS)
def test_tv_denoises_piecewise_constant(denoise):
    clean, noisy = _square()
    out = denoise(noisy, 0.05, (0, 1))
    assert out.shape == noisy.shape
    assert _tv(out) < 0.6 * _tv(noisy)
    assert _err(out, clean) < 0.6 * _err(noisy, clean)


@pytest.mark.parametrize("denoise", TV_DENOISERS)
def test_tv_constant_is_fixed_point(denoise):
    c = torch.full((N, N), 0.7 + 0.2j, dtype=torch.complex64)
    torch.testing.assert_close(denoise(c, 0.1, (0, 1)), c, atol=1e-5, rtol=0)


@pytest.mark.parametrize("denoise", TV_DENOISERS)
def test_tv_preserves_mean(denoise):
    # Periodic differences annihilate constants, so the data term alone fixes the mean.
    _, noisy = _square()
    for lam in (0.05, 1000.0):
        out = denoise(noisy, lam, (0, 1))
        assert abs(out.mean() - noisy.mean()).item() < 1e-5
    assert out.std() < 0.8 * noisy.std()


def test_rof_acts_only_along_axes():
    # Rows differ, columns are constant: TV along the last axis sees nothing.
    img = torch.zeros(N, N, dtype=torch.complex64)
    img[N // 4 : 3 * N // 4, :] = 1
    torch.testing.assert_close(rof(img, 0.5, -1), img, atol=1e-5, rtol=0)
    # Along axis 0 each level of the width-8 step moves by about 2 * lamda / 8.
    assert (rof(img, 0.5, 0) - img).abs().max() > 0.05


def test_rof_three_dimensional_batch():
    clean, noisy = _square()
    batch = torch.stack([noisy, clean])
    out = rof(batch, 0.05, (1, 2))
    assert out.shape == batch.shape
    # TV rounds the square's corners by about lamda; the clean frame is otherwise kept.
    torch.testing.assert_close(out[1], clean, atol=0.1, rtol=0)
    assert _err(out[0], clean) < 0.6 * _err(noisy, clean)


def test_tgv_tvscales_follow_axes():
    img = torch.zeros(N, N, dtype=torch.complex64)
    img[N // 4 : 3 * N // 4, :] = 1
    y = img + _noise((N, N), 0.05, 1)
    out = tgv(y, 0.1, (0, 1), tvscales=(1e-3, 1.0))
    # Axis 1 carries the weight, axis 0 almost none.
    assert _tv(out, (1,)) < 0.1 * _tv(y, (1,))
    assert _tv(out, (0,)) > 0.5 * _tv(y, (0,))


def test_tgv_rejects_mismatched_tvscales():
    with pytest.raises(ValueError, match="one value per axis"):
        tgv(torch.zeros(4, 4, dtype=torch.complex64), 0.1, (0, 1), tvscales=(1.0,))


def test_nlmeans_constant_unchanged():
    c = torch.full((N, N), 0.7 + 0.2j, dtype=torch.complex64)
    torch.testing.assert_close(nlmeans(c, (0, 1)), c, atol=1e-6, rtol=0)


def test_nlmeans_reduces_noise():
    clean, noisy = _square()
    out = nlmeans(noisy, (0, 1), h=0.2)
    assert _err(out, clean) < 0.6 * _err(noisy, clean)


def test_nlmeans_zero_distance_is_identity():
    _, noisy = _square()
    torch.testing.assert_close(nlmeans(noisy, (0, 1), patch_distance=0), noisy)


def test_nlmeans_large_h_is_box_mean():
    y = _noise((N, N), 1.0, 3)
    d = 2
    out = nlmeans(y, (0, 1), patch_length=3, patch_distance=d, h=1e4)
    box = torch.nn.functional.avg_pool2d(
        torch.view_as_real(y).permute(2, 0, 1)[None], 2 * d + 1, stride=1
    )[0].permute(1, 2, 0)
    torch.testing.assert_close(out[d:-d, d:-d], torch.view_as_complex(box.contiguous()))


def test_nlmeans_rejects_even_patch():
    with pytest.raises(ValueError, match="odd"):
        nlmeans(torch.zeros(4, 4, dtype=torch.complex64), (0, 1), patch_length=4)
