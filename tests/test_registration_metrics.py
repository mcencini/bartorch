"""Registration and image-quality metrics, held against their definitions.

The metrics are compared with numpy written from the textbook formula, the
registrations with a displacement put in by hand, and the displacement field
with torch's own resampling, so none of them is BART checked against BART.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from bartorch.interp import affine_transform, warp
from bartorch.metrics import mse, nrmse, psnr, roi_stat, ssim
from bartorch.registration import estimate_shift, register_affine, register_nonrigid

N = 32


def _gaussian(n: int, cy: float, cx: float, s: float) -> torch.Tensor:
    y = torch.arange(n) - n // 2
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    return torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))


def _phantom(n: int = N, dy: float = 0.0, dx: float = 0.0) -> torch.Tensor:
    """Two smooth blobs, displaced by ``(dy, dx)`` voxels."""
    img = _gaussian(n, dy, dx, 4.0) + 0.5 * _gaussian(n, dy + 5, dx - 6, 2.5)
    return img.to(torch.complex64)


def _pair(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)
    noise = torch.randn(*shape, generator=g) + 1j * torch.randn(*shape, generator=g)
    return ref.to(torch.complex64), (ref + 0.2 * noise).to(torch.complex64)


# --- metrics ------------------------------------------------------------------


def test_nrmse():
    ref, x = _pair((16, 16))
    r, xn = ref.numpy(), x.numpy()
    assert nrmse(ref, x) == pytest.approx(np.linalg.norm(xn - r) / np.linalg.norm(r), rel=1e-5)

    x = ((0.7 - 0.4j) * ref + 0.1 * (x - ref)).to(torch.complex64)
    xn = x.numpy()
    c = np.vdot(r, xn) / np.vdot(r, r).real
    want = np.linalg.norm(xn - c * r) / np.linalg.norm(c * r)
    assert nrmse(ref, x, scaled=True) == pytest.approx(want, rel=1e-5)


def test_mse():
    ref, x = _pair((3, 16, 16))
    r, xn = ref.numpy(), x.numpy()
    assert mse(ref, x) == pytest.approx(np.mean(np.abs(xn - r) ** 2), rel=1e-5)
    want = np.mean((np.abs(xn) - np.abs(r)) ** 2)
    assert mse(ref, x, magnitude=True) == pytest.approx(want, rel=1e-5)


def test_mse_magnitude_is_rss_over_coils():
    ref, x = _pair((4, 1, 16, 16))  # (coils, z, y, x)
    r, xn = ref.numpy(), x.numpy()
    rss = lambda a: np.sqrt(np.sum(np.abs(a) ** 2, axis=0))  # noqa: E731
    assert mse(ref, x, magnitude=True) == pytest.approx(np.mean((rss(xn) - rss(r)) ** 2), rel=1e-5)


def test_psnr_is_mean_over_images():
    # Two images along an axis ahead of (coils, z, y, x), of different peak.
    ref, x = _pair((2, 1, 1, 16, 16))
    ref[1] *= 3
    r, xn = np.abs(ref.numpy()), np.abs(x.numpy())
    per_image = [
        20 * math.log10(r[i].max()) - 10 * math.log10(np.mean((xn[i] - r[i]) ** 2))
        for i in range(2)
    ]
    assert psnr(ref, x) == pytest.approx(np.mean(per_image), rel=1e-5)


def _ssim_numpy(ref: np.ndarray, x: np.ndarray, win: int = 7) -> float:
    """SSIM of magnitudes over uniform windows, both divided by max|ref|."""
    peak = np.abs(ref).max()
    a, b = np.abs(x) / peak, np.abs(ref) / peak
    view = lambda v: np.lib.stride_tricks.sliding_window_view(v, (win, win))  # noqa: E731
    mean = lambda v: view(v).mean(axis=(-2, -1))  # noqa: E731
    ma, mb = mean(a), mean(b)
    va, vb, cov = mean(a * a) - ma**2, mean(b * b) - mb**2, mean(a * b) - ma * mb
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))
    return float(s.mean())


def test_ssim():
    ref = _phantom(24)
    g = torch.Generator().manual_seed(1)
    x = (ref + 0.05 * torch.randn(24, 24, generator=g)).to(torch.complex64)
    assert ssim(ref, x) == pytest.approx(_ssim_numpy(ref.numpy(), x.numpy()), abs=1e-4)
    assert ssim(ref, ref) == pytest.approx(1.0, abs=1e-5)


def _energy(v):
    return (np.abs(v - v.mean(-1, keepdims=True)) ** 2).sum(-1)


ROI_STATS = {
    "count": lambda v, b: np.full(v.shape[0], v.shape[1], dtype=float),
    "sum": lambda v, b: v.sum(-1),
    "mean": lambda v, b: v.mean(-1),
    "energy": lambda v, b: _energy(v),
    "variance": lambda v, b: _energy(v) / (v.shape[1] - b),
    "std": lambda v, b: np.sqrt(_energy(v) / (v.shape[1] - b)),
}


@pytest.mark.parametrize(
    ("stat", "bessel"),
    [(s, False) for s in ROI_STATS] + [("variance", True), ("std", True)],
)
def test_roi_stat(stat, bessel):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 8, 8, generator=g) + 1j * torch.randn(3, 8, 8, generator=g)
    x = x.to(torch.complex64)
    roi = torch.zeros(8, 8)
    roi[2:6, 1:5] = 1
    got = roi_stat(roi, x, stat, bessel=bessel)
    assert got.shape == (3, 1, 1)
    values = x.numpy()[:, roi.bool().numpy()]
    want = ROI_STATS[stat](values, 1 if bessel else 0)
    np.testing.assert_allclose(got.numpy().reshape(3), want, rtol=1e-5, atol=1e-5)


def test_roi_stat_regions_along_an_axis():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(8, 8, generator=g).to(torch.complex64)
    roi = torch.zeros(2, 8, 8)
    roi[0, :4] = 1
    roi[1, 4:] = 1
    got = roi_stat(roi, x)
    assert got.shape == (2, 1, 1)
    want = [x[:4].real.mean(), x[4:].real.mean()]
    np.testing.assert_allclose(got.real.reshape(2).numpy(), want, rtol=1e-5)


# --- registration -------------------------------------------------------------


def test_estimate_shift_recovers_roll():
    a = _phantom()
    b = torch.roll(a, shifts=(2, -3), dims=(0, 1))
    shift = estimate_shift(a, b, (0, 1))
    np.testing.assert_allclose(shift.numpy(), [-2, 3], atol=1e-3)
    whole = tuple(int(round(s)) for s in shift.tolist())
    assert torch.equal(torch.roll(b, shifts=whole, dims=(0, 1)), a)
    np.testing.assert_allclose(estimate_shift(a, b, (1, 0)).numpy(), [3, -2], atol=1e-3)
    fov = estimate_shift(a, b, (-2, -1), fov_units=True)
    np.testing.assert_allclose(fov.numpy(), [-2 / N, 3 / N], atol=1e-4)


@pytest.mark.parametrize("transform", ["translation", "rigid"])
def test_register_affine_recovers_translation(transform):
    dy, dx = 2.0, -3.0
    ref, moved = _phantom(), _phantom(dy=dy, dx=dx)
    matrix = register_affine(ref, moved, transform=transform)
    assert matrix.shape == (2, 3)
    # Translation over (y, x), in field-of-view units.
    np.testing.assert_allclose(matrix[:, 2].numpy(), [dy / N, dx / N], atol=0.01)
    np.testing.assert_allclose(matrix[:, :2].numpy(), np.eye(2), atol=0.05)
    # The contract with the resampler: moved is brought back onto reference.
    aligned = affine_transform(moved, matrix, (-2, -1))
    assert (aligned - ref).norm() < 0.05 * (moved - ref).norm()


def test_register_affine_3d_translation():
    n, d = 16, (1.0, 2.0, -1.5)  # (dz, dy, dx) voxels
    zz, yy, xx = torch.meshgrid(*(torch.arange(n) - n // 2,) * 3, indexing="ij")

    def volume(dz=0.0, dy=0.0, dx=0.0):
        r2 = (zz - dz) ** 2 + (yy - dy) ** 2 + ((xx - dx) / 1.5) ** 2
        return torch.exp(-r2 / 8).to(torch.complex64)

    ref, moved = volume(), volume(*d)
    matrix = register_affine(ref, moved, transform="translation")
    assert matrix.shape == (3, 4)
    np.testing.assert_allclose(matrix[:, 3].numpy(), np.array(d) / n, atol=0.01)
    aligned = affine_transform(moved, matrix, (-3, -2, -1))
    assert (aligned - ref).norm() < 0.1 * (moved - ref).norm()


def test_register_affine_needs_both_masks():
    img = _phantom()
    with pytest.raises(ValueError, match="both masks"):
        register_affine(img, img, reference_mask=torch.ones(N, N))


def _warp(image: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """``image(p + field(p))``, bilinear, ``field`` in voxels over ``(y, x)``."""
    n_y, n_x = image.shape
    yy, xx = torch.meshgrid(torch.arange(n_y), torch.arange(n_x), indexing="ij")
    py, px = yy + field[..., 0], xx + field[..., 1]
    grid = torch.stack([2 * px / (n_x - 1) - 1, 2 * py / (n_y - 1) - 1], dim=-1)[None]
    return F.grid_sample(image[None, None], grid.float(), align_corners=True)[0, 0]


@pytest.mark.parametrize("optical_flow", [False, True])
def test_register_nonrigid_aligns(optical_flow):
    dy, dx = 1.5, -2.0
    ref, moved = _phantom(), _phantom(dy=dy, dx=dx)
    field, inverse = register_nonrigid(ref, moved, (0, 1), optical_flow=optical_flow)
    assert field.shape == (N, N, 2)
    centre = (slice(12, 20), slice(12, 20))
    np.testing.assert_allclose(field[centre].mean(dim=(0, 1)).numpy(), [dy, dx], atol=0.6)

    before = (moved.real - ref.real).norm()
    assert (_warp(moved.real, field) - ref.real).norm() < before / 4
    # The contract with the resampler: warp takes the field as returned.
    assert (warp(moved, field, (0, 1)) - ref).norm() < before / 4
    if optical_flow:
        assert inverse is None
    else:
        assert inverse.shape == (N, N, 2)
        assert (_warp(ref.real, inverse) - moved.real).norm() < before / 4
