"""Wavelet transforms and thresholds, held against PyWavelets and closed forms."""

import warnings

import numpy as np
import pytest
import torch

import bartorch
from bartorch.thresh import hard_thresh, soft_thresh
from bartorch.wavelet import fwt, iwt

pywt = pytest.importorskip("pywt")

#: PyWavelets' name for each of BART's ``wavelet`` filters.
PYWT_NAMES = {"haar": "haar", "dau2": "db2", "cdf44": "bior4.4"}


def _crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.complex64, generator=g)


def _inner(a: torch.Tensor, b: torch.Tensor) -> complex:
    return complex(torch.vdot(a.flatten().to(torch.complex128), b.flatten().to(torch.complex128)))


def _wavedec_like_bart(x: np.ndarray, wavelet: str) -> np.ndarray:
    """PyWavelets' ``wavedec`` down to where BART stops: a half-length under 16."""
    name = PYWT_NAMES[wavelet]
    taps = pywt.Wavelet(name).dec_len
    n, levels = x.shape[-1], 1
    while (n + taps - 1) // 2 >= 16:
        n, levels = (n + taps - 1) // 2, levels + 1
    with warnings.catch_warnings():
        # PyWavelets warns when boundary effects reach every coefficient.
        warnings.simplefilter("ignore", UserWarning)
        coeffs = pywt.wavedec(x, name, mode="symmetric", level=levels, axis=-1)
    return np.concatenate(coeffs, axis=-1)


@pytest.mark.parametrize("wavelet", sorted(PYWT_NAMES))
@pytest.mark.parametrize("n", [16, 64])
def test_one_axis_is_pywavelets_wavedec_in_symmetric_mode(wavelet, n):
    x = _crandn(3, n)
    got = fwt(x, -1, wavelet=wavelet).numpy()
    ref = _wavedec_like_bart(x.numpy(), wavelet)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_one_cdf97_level_is_bior44_periodization_with_the_detail_negated():
    """At 32 samples BART stops after one level."""
    x = _crandn(2, 32)
    got = fwt(x, -1, wavelet="cdf97").numpy()
    approx, detail = pywt.dwt(x.numpy(), "bior4.4", mode="periodization", axis=-1)
    np.testing.assert_allclose(got, np.concatenate([approx, -detail], axis=-1), atol=1e-5)


CASES = [
    ((3, 20, 24), (-2, -1)),
    ((20, 3), 0),
    ((2, 17, 16, 18), (1, 3)),
]


@pytest.mark.parametrize("wavelet", ["haar", "dau2", "cdf44", "cdf97"])
@pytest.mark.parametrize("shape,axes", CASES)
def test_iwt_inverts_fwt(wavelet, shape, axes):
    x = _crandn(*shape)
    coeffs = fwt(x, axes, wavelet=wavelet)
    back = iwt(coeffs, shape, axes, wavelet=wavelet)
    assert back.shape == x.shape
    torch.testing.assert_close(back, x, rtol=1e-4, atol=1e-4)


def test_coefficients_collect_in_the_last_transformed_axis():
    coeffs = fwt(_crandn(2, 20, 3, 24), (1, 3), wavelet="dau2")
    assert coeffs.shape[:3] == (2, 1, 3)


def test_haar_on_even_lengths_is_orthogonal():
    x = _crandn(2, 32, 40)
    y = _crandn(2, 1, 32 * 40, seed=1)
    wx = fwt(x, (-2, -1), wavelet="haar")
    assert wx.shape == y.shape
    lhs, rhs = _inner(wx, y), _inner(x, iwt(y, x.shape, (-2, -1), wavelet="haar"))
    assert abs(lhs - rhs) < 1e-4 * abs(lhs)
    norm = float(torch.linalg.vector_norm(x))
    assert float(torch.linalg.vector_norm(wx)) == pytest.approx(norm, rel=1e-5)


@pytest.mark.parametrize("wavelet,n", [("dau2", 40), ("cdf44", 40), ("haar", 34)])
def test_synthesis_is_not_the_adjoint_of_a_redundant_transform(wavelet, n):
    x = _crandn(4, n)
    wx = fwt(x, -1, wavelet=wavelet)
    assert wx.shape[-1] > n
    y = _crandn(*wx.shape, seed=1)
    lhs, rhs = _inner(wx, y), _inner(x, iwt(y, x.shape, -1, wavelet=wavelet))
    assert abs(lhs - rhs) > 1e-2 * abs(lhs)


def test_sizes_bart_would_assert_on_are_refused():
    with pytest.raises(ValueError, match=">= 16"):
        fwt(_crandn(8, 32), (-2, -1))
    coeffs = fwt(_crandn(3, 32), -1)
    with pytest.raises(ValueError, match="coefficients"):
        iwt(coeffs[..., :-1], (3, 32), -1)
    with pytest.raises(ValueError, match="wavelet must be"):
        fwt(_crandn(32), -1, wavelet="db4")


def _soft_reference(x: torch.Tensor, lamda: float, joint=()) -> torch.Tensor:
    x = x.to(torch.complex128)
    norm = x.abs() if not joint else torch.linalg.vector_norm(x, dim=joint, keepdim=True)
    scale = torch.clamp(1 - lamda / norm, min=0)
    return (x * scale).to(torch.complex64)


def test_soft_thresh_is_its_closed_form():
    x = _crandn(4, 5, 6)
    torch.testing.assert_close(soft_thresh(0.7, x), _soft_reference(x, 0.7), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("joint", [(0,), (-2, -1), 1])
def test_joint_soft_thresh_shrinks_by_the_norm_over_the_joint_axes(joint):
    x = _crandn(3, 4, 5)
    dims = (joint,) if isinstance(joint, int) else joint
    torch.testing.assert_close(
        soft_thresh(1.5, x, joint_axes=joint), _soft_reference(x, 1.5, dims), rtol=1e-5, atol=1e-6
    )


def test_hard_thresh_keeps_what_exceeds_the_threshold():
    x = _crandn(1, 6, 7)
    ref = torch.where(x.abs() > 0.8, x, torch.zeros_like(x))
    got = hard_thresh(0.8, x)
    assert got.shape == x.shape
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_the_wrappers_are_exported_at_the_top_level():
    assert (bartorch.fwt, bartorch.iwt) == (fwt, iwt)
    assert (bartorch.soft_thresh, bartorch.hard_thresh) == (soft_thresh, hard_thresh)
