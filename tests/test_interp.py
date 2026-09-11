"""Interpolation, warps, affine resampling and FoV shifts, held against numpy and scipy.

Each case pins one convention BART does not state on the command line: the
unit of a position (voxel index or field of view), the order of a field's
components, the direction of a mapping and the sign of a phase.
"""

import numpy as np
import pytest
import torch

import bartorch.tools as bt
from bartorch.interp import affine_transform, fovshift, interpolate, warp

scipy_ndimage = pytest.importorskip("scipy.ndimage")


def _randn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.complex64, generator=g)


def _map_coordinates(img: np.ndarray, coords: np.ndarray, order: int) -> np.ndarray:
    """scipy's resampling of a complex array, zero outside the grid."""

    def real(a):
        return scipy_ndimage.map_coordinates(a, coords, order=order, mode="grid-constant")

    return real(img.real) + 1j * real(img.imag)


# -- interpolate ------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 3])
def test_integer_positions_return_the_samples(order):
    img = _randn(6, 8)
    iy, ix = torch.meshgrid(torch.arange(6.0), torch.arange(8.0), indexing="ij")
    coord = torch.stack([iy, ix], dim=-1)
    out = interpolate(img, coord, axes=(-2, -1), order=order)
    torch.testing.assert_close(out, img, rtol=1e-5, atol=1e-5)


def test_linear_at_half_positions_is_the_mean_of_the_neighbours():
    img = _randn(6, 8)
    ix = torch.arange(7.0) + 0.5
    coord = torch.stack([torch.full_like(ix, 2.0), ix], dim=-1)[None]  # (1, 7, 2)
    out = interpolate(img, coord, axes=(-2, -1))
    torch.testing.assert_close(out[0], (img[2, :-1] + img[2, 1:]) / 2)


def test_cubic_uses_keys_weights():
    """Keys' kernel, a = -1/2, weighs the neighbours of a half position (-1, 9, 9, -1)/16."""
    img = _randn(1, 8)
    out = interpolate(img, torch.tensor([[[0.0, 3.5]]]), axes=(-2, -1), order=3)
    ref = (-img[0, 2] + 9 * img[0, 3] + 9 * img[0, 4] - img[0, 5]) / 16
    torch.testing.assert_close(out.reshape(()), ref)


@pytest.mark.parametrize("order", [0, 1])
def test_scattered_positions_match_scipy(order):
    img = _randn(9, 11)
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(20, 2, generator=g) * torch.tensor([7.0, 9.0]) + 0.6
    coord = pos[:, None, :]  # 20 points on the first axis, one on the second
    out = interpolate(img, coord, axes=(-2, -1), order=order)
    ref = _map_coordinates(img.numpy(), pos.T.numpy().astype(np.float64), order)
    np.testing.assert_allclose(out[:, 0].numpy(), ref, rtol=1e-4, atol=1e-5)


def test_the_components_follow_the_order_of_axes():
    img = _randn(6, 8)
    yx = interpolate(img, torch.tensor([[[2.0, 3.5]]]), axes=(-2, -1))
    xy = interpolate(img, torch.tensor([[[3.5, 2.0]]]), axes=(-1, -2))
    torch.testing.assert_close(xy, yx)
    torch.testing.assert_close(yx.reshape(()), (img[2, 3] + img[2, 4]) / 2)


def test_positions_outside_the_grid_see_zeros():
    img = _randn(1, 8)
    out = interpolate(
        img, torch.tensor([[[0.0, -0.5], [0.0, 7.5]]]).reshape(1, 2, 2), axes=(-2, -1)
    )
    torch.testing.assert_close(out[0], torch.stack([img[0, 0], img[0, 7]]) / 2)


def test_positions_are_shared_over_an_axis_not_interpolated():
    img = _randn(3, 6, 8)
    coord = torch.tensor([[1.0, 2.5], [4.0, 6.0]])[:, None, :]  # (2, 1, 2)
    out = interpolate(img, coord, axes=(-2, -1))
    assert out.shape == (3, 2, 1)
    torch.testing.assert_close(out[:, 0, 0], (img[:, 1, 2] + img[:, 1, 3]) / 2)
    torch.testing.assert_close(out[:, 1, 0], img[:, 4, 6])


def test_a_single_axis_of_a_volume():
    img = _randn(4, 5, 6)
    coord = (torch.arange(5.0)[:, None] + 0.25).expand(5, 6)[..., None]  # (5, 6, 1)
    out = interpolate(img, coord, axes=-2)
    ref = 0.75 * img[:, :, :] + 0.25 * torch.cat([img[:, 1:], torch.zeros_like(img[:, :1])], 1)
    torch.testing.assert_close(out, ref)


def test_an_unsupported_order_is_refused():
    with pytest.raises(ValueError, match="order"):
        interpolate(_randn(4, 4), torch.zeros(1, 1, 2), axes=(-2, -1), order=2)


# -- warp -------------------------------------------------------------------


def test_zero_displacement_is_the_identity():
    img = _randn(6, 8)
    torch.testing.assert_close(warp(img, torch.zeros(6, 8, 2), axes=(-2, -1)), img)


def test_an_integer_displacement_pulls_from_the_displaced_voxel():
    img = _randn(6, 8)
    disp = torch.zeros(6, 8, 2)
    disp[..., 0] = 2.0  # along y
    disp[..., 1] = -1.0  # along x
    out = warp(img, disp, axes=(-2, -1))
    torch.testing.assert_close(out[:-2, 1:], img[2:, :-1])
    assert out[-2:].abs().max() == 0 and out[:, 0].abs().max() == 0


def test_a_smooth_displacement_matches_scipy():
    img = _randn(3, 10, 12)
    iy, ix = torch.meshgrid(torch.arange(10.0), torch.arange(12.0), indexing="ij")
    disp = torch.stack([0.7 * torch.sin(ix / 3), 0.9 * torch.cos(iy / 4)], dim=-1)
    out = warp(img, disp, axes=(-2, -1))
    coords = torch.stack([iy + disp[..., 0], ix + disp[..., 1]]).numpy().astype(np.float64)
    for b in range(3):
        ref = _map_coordinates(img[b].numpy(), coords, 1)
        np.testing.assert_allclose(out[b].numpy(), ref, rtol=1e-4, atol=1e-5)


def test_displacement_components_follow_the_order_of_axes():
    img = _randn(6, 8)
    disp = torch.zeros(6, 8, 2)
    disp[..., 0] = 1.0
    torch.testing.assert_close(warp(img, disp, axes=(-1, -2))[:, :-1], img[:, 1:])


def test_warp_refuses_axes_that_are_not_trailing():
    with pytest.raises(ValueError, match="last"):
        warp(_randn(4, 5, 6), torch.zeros(4, 5, 6, 2), axes=(0, 1))


# -- affine_transform -------------------------------------------------------


def test_the_identity_matrix_returns_the_input():
    img = _randn(6, 8)
    out = affine_transform(img, torch.tensor([[1.0, 0, 0], [0, 1, 0]]), axes=(-2, -1))
    torch.testing.assert_close(out, img)


def test_a_translation_is_in_fields_of_view_and_maps_output_to_input():
    img = _randn(5, 6, 8)
    m = torch.eye(4)
    m[0, 3] = 1 / 5  # one voxel along z
    m[2, 3] = -2 / 8  # minus two voxels along x
    out = affine_transform(img, m, axes=(-3, -2, -1))
    torch.testing.assert_close(out[:-1, :, 2:], img[1:, :, :-2])


def test_the_matrix_rows_follow_the_order_of_axes():
    img = _randn(8, 8)
    swap = torch.tensor([[0.0, 1, 0], [1, 0, 0]])
    torch.testing.assert_close(affine_transform(img, swap, axes=(-2, -1)), img.T)
    shift_x = torch.tensor([[1.0, 0, 1 / 8], [0, 1, 0]])
    torch.testing.assert_close(affine_transform(img, shift_x, axes=(-1, -2))[:, :-1], img[:, 1:])


def test_oshape_resamples_the_whole_field_of_view():
    img = _randn(4, 6)
    out = affine_transform(img, torch.tensor([[1.0, 0, 0], [0, 1, 0]]), (-2, -1), oshape=(8, 12))
    assert out.shape == (8, 12)
    torch.testing.assert_close(out[::2, ::2], img)
    torch.testing.assert_close(out[::2, 1:-1:2], (img[:, :-1] + img[:, 1:]) / 2)


def test_a_homogeneous_matrix_must_end_in_the_unit_row():
    with pytest.raises(ValueError, match="last row"):
        affine_transform(_randn(4, 4), torch.ones(3, 3), axes=(-2, -1))


# -- fovshift ---------------------------------------------------------------


def _kspace(img: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))


def _image(ksp: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ksp)))


@pytest.mark.parametrize("pixels", [True, False])
def test_a_cartesian_shift_moves_the_image_by_minus_shift(pixels):
    img = _randn(8, 16, 12, seed=3).numpy()
    shift_vox = (1, -3, 2)
    shift = shift_vox if pixels else tuple(s / n for s, n in zip(shift_vox, img.shape))
    out = fovshift(torch.from_numpy(_kspace(img)).to(torch.complex64), shift, pixels=pixels)
    ref = np.roll(img, [-s for s in shift_vox], axis=(0, 1, 2))
    np.testing.assert_allclose(_image(out.numpy()), ref, atol=1e-4)


def test_a_two_dimensional_shift_leaves_the_leading_axes():
    img = _randn(3, 8, 10, seed=4).numpy()
    ksp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, (-2, -1))), (-2, -1))
    out = fovshift(torch.from_numpy(ksp).to(torch.complex64), (2, -1), pixels=True).numpy()
    back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(out, (-2, -1))), (-2, -1))
    np.testing.assert_allclose(back, np.roll(img, (-2, 1), axis=(1, 2)), atol=1e-4)


def test_a_non_cartesian_shift_is_the_phase_of_the_trajectory():
    traj = bt.traj(x=8, y=6, radial=True)  # (spokes, samples, 3), (kx, ky, kz)
    ksp = _randn(*traj.shape[:-1], 1, seed=5)
    shift = (0.1, -0.25)  # (y, x) in fields of view
    out = fovshift(ksp, shift, traj=traj)
    k = traj.real.numpy().astype(np.float64)
    phase = np.exp(2j * np.pi * (k[..., 0] * shift[1] + k[..., 1] * shift[0]))
    np.testing.assert_allclose(out.numpy(), ksp.numpy() * phase[..., None], rtol=1e-4, atol=1e-5)


def test_pixels_with_a_trajectory_is_refused():
    traj = bt.traj(x=8, y=4)
    with pytest.raises(ValueError, match="Cartesian"):
        fovshift(_randn(4, 8, 1), (1, 1), traj=traj, pixels=True)
