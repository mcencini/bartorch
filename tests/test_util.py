"""The array utilities in :mod:`bartorch.util`, each held against numpy or scipy."""

import math

import numpy as np
import pytest
import scipy.ndimage as ndi
import torch
from numpy.lib.stride_tricks import sliding_window_view

import bartorch
from bartorch import util

RNG = np.random.default_rng(0)


def _complex(*shape) -> torch.Tensor:
    return torch.from_numpy(
        (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)).astype(np.complex64)
    )


def _real(a) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(torch.complex64)


def _resize_reference(a: np.ndarray, oshape, anchor: str) -> np.ndarray:
    out = a
    for axis, want in enumerate(oshape):
        have = out.shape[axis]
        if have == want:
            continue
        if anchor == "start":
            offset = 0
        elif anchor == "front":
            offset = abs(want - have)
        else:
            offset = abs(want // 2 - have // 2)
        index = [slice(None)] * out.ndim
        if want < have:
            index[axis] = slice(offset, offset + want)
            out = out[tuple(index)]
        else:
            shape = list(out.shape)
            shape[axis] = want
            padded = np.zeros(shape, dtype=out.dtype)
            index[axis] = slice(offset, offset + have)
            padded[tuple(index)] = out
            out = padded
    return out


@pytest.mark.parametrize("anchor", ["center", "front", "start"])
@pytest.mark.parametrize("oshape", [(3, 9), (8, 4), (5, 6), (1, 6)])
def test_resize(anchor, oshape):
    x = _complex(5, 6)
    out = util.resize(x, oshape, anchor=anchor)
    assert out.shape == oshape
    np.testing.assert_array_equal(out.numpy(), _resize_reference(x.numpy(), oshape, anchor))


def test_circshift_is_torch_roll():
    x = _complex(1, 4, 5, 6)
    out = util.circshift(x, (2, -3, 7), (1, -1, 2))
    torch.testing.assert_close(out, torch.roll(x, (2, -3, 7), (1, -1, 2)), rtol=0, atol=0)


def _cyclic_conv(x: np.ndarray, k: np.ndarray, axes) -> np.ndarray:
    """``out[i] = sum_j k[j] x[(i - j + (len_k - 1) // 2) % n]`` along ``axes``, by FFT."""
    k = k.reshape((1,) * (x.ndim - k.ndim) + k.shape)
    padded = np.zeros(
        tuple(x.shape[a] if a in axes else k.shape[a] for a in range(x.ndim)), complex
    )
    padded[tuple(slice(0, n) for n in k.shape)] = k
    padded = np.roll(padded, [-((k.shape[a] - 1) // 2) for a in axes], axis=axes)
    return np.fft.ifftn(np.fft.fftn(x, axes=axes) * np.fft.fftn(padded, axes=axes), axes=axes)


@pytest.mark.parametrize("kshape", [(3, 3), (2, 4), (1, 5), (7,)])
def test_conv(kshape):
    x, k = _complex(6, 7), _complex(*kshape)
    axes = (0, 1) if len(kshape) == 2 else (1,)
    out = util.conv(x, k, axes if len(axes) > 1 else -1)
    np.testing.assert_allclose(out.numpy(), _cyclic_conv(x.numpy(), k.numpy(), axes), atol=1e-5)


def test_conv_per_slice_kernel():
    x, k = _complex(3, 8), _complex(3, 3)
    out = util.conv(x, k, -1)
    ref = np.stack([_cyclic_conv(x.numpy()[i], k.numpy()[i], (0,)) for i in range(3)])
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-5)


@pytest.mark.parametrize("hann", [False, True])
def test_window(hann):
    x = _complex(2, 5, 8)
    w = np.hanning if hann else np.hamming
    out = util.window(x, (1, 2), hann=hann)
    ref = x.numpy() * w(5)[:, None] * w(8)[None, :]
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)
    out = util.window(x, 1, hann=hann)
    np.testing.assert_allclose(out.numpy(), x.numpy() * w(5)[:, None], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("length", [1, 3, 4, 9])
def test_median_filter_real(length):
    a = RNG.uniform(0.1, 1.0, (4, 9))
    out = util.median_filter(_real(a), -1, length)
    ref = np.median(sliding_window_view(a, length, axis=-1), axis=-1)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out.real.numpy(), ref, rtol=1e-6)


def test_median_filter_complex_is_median_magnitude():
    x = _complex(7, 3)
    out = util.median_filter(x, 0, 5)
    windows = sliding_window_view(x.numpy(), 5, axis=0)  # (3, 3, 5)
    order = np.argsort(np.abs(windows), axis=-1)
    ref = np.take_along_axis(windows, order[..., 2:3], axis=-1)[..., 0]
    np.testing.assert_array_equal(out.numpy(), ref)


def test_geometric_median_of_a_square():
    # Four corners of a square about c: the geometric median is c.
    c = 0.7 + 0.4j
    corners = c + np.array([1, 1j, -1, -1j]) * 0.5
    out = util.median_filter(torch.from_numpy(corners.astype(np.complex64)), 0, 4, geometric=True)
    assert out.shape == (1,)
    assert abs(complex(out[0]) - c) < 1e-3


@pytest.mark.parametrize("length", [1, 2, 5])
def test_moving_average(length):
    x = _complex(5, 3)
    out = util.moving_average(x, 0, length)
    ref = sliding_window_view(x.numpy(), length, axis=0).mean(axis=-1)
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)


def test_filter_length_is_checked():
    with pytest.raises(ValueError):
        util.moving_average(_complex(3, 4), -1, 5)


@pytest.mark.parametrize("axes", [-1, (0, 2)])
@pytest.mark.parametrize("l1", [False, True])
def test_normalize(axes, l1):
    x = _complex(3, 4, 5)
    a = x.numpy()
    norm = (
        np.abs(a).sum(axis=axes, keepdims=True)
        if l1
        else np.sqrt((np.abs(a) ** 2).sum(axis=axes, keepdims=True))
    )
    np.testing.assert_allclose(util.normalize(x, axes, l1=l1).numpy(), a / norm, rtol=1e-5)


def test_mip():
    a = RNG.uniform(0.1, 1.0, (3, 4, 5))
    x = _real(a)
    np.testing.assert_allclose(util.mip(x, 0).real.numpy(), a.max(axis=0, keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(
        util.mip(x, (0, 2), minimum=True).real.numpy(), a.min(axis=(0, 2), keepdims=True), rtol=1e-6
    )
    z = _complex(3, 4, 5)
    np.testing.assert_allclose(
        util.mip(z, -1, magnitude=True).numpy(),
        np.abs(z.numpy()).max(axis=-1, keepdims=True),
        rtol=1e-6,
    )


def test_mip_takes_real_parts_and_starts_from_zero():
    z = torch.tensor([[1 + 5j, 2 + 0j], [3 + 0j, 0.5 - 1j]], dtype=torch.complex64)
    torch.testing.assert_close(util.mip(z, 0), torch.tensor([[3, 2]], dtype=torch.complex64))
    negative = _real([[-3, -1], [-2, -5]])
    torch.testing.assert_close(util.mip(negative, 0), torch.zeros(1, 2, dtype=torch.complex64))
    torch.testing.assert_close(util.mip(negative, 0, minimum=True), _real([[-3, -5]]))


@pytest.mark.parametrize("bound", [math.pi, 0.5])
def test_unwrap(bound):
    period = 2 * bound
    t = np.linspace(0, 1, 40)
    phase = np.stack([6 * period * t**2, -4 * period * t])  # (2, 40)
    wrapped = (phase + bound) % period - bound
    out = util.unwrap(_real(wrapped), -1, bound=bound)
    ref = np.unwrap(wrapped, period=period, axis=-1)
    np.testing.assert_allclose(out.real.numpy(), ref, atol=1e-4)
    np.testing.assert_allclose(ref, phase, atol=1e-4)


def test_unwrap_leaves_the_imaginary_part():
    wrapped = np.angle(np.exp(1j * np.linspace(0, 12, 30)))
    x = _real(wrapped) + 2j
    out = util.unwrap(x, 0)
    np.testing.assert_allclose(out.real.numpy(), np.unwrap(wrapped), atol=1e-4)
    np.testing.assert_allclose(out.imag.numpy(), 2.0)


def test_casorati():
    x = _complex(5, 6)
    out = util.casorati(x, (2, 3), (0, 1))
    ref = sliding_window_view(x.numpy(), (2, 3)).reshape(-1, 6).T
    np.testing.assert_array_equal(out.numpy(), ref)


def test_casorati_takes_other_axes_whole():
    x = _complex(2, 5, 6)
    out = util.casorati(x, 4, -1)
    ref = sliding_window_view(x.numpy(), (2, 5, 4)).reshape(-1, 2 * 5 * 4).T
    assert out.shape == (40, 3)
    np.testing.assert_array_equal(out.numpy(), ref)


def _footprint(size: int, ndim: int, ball: bool) -> np.ndarray:
    if not ball:
        return np.ones((size,) * ndim, bool)
    r = size // 2
    grid = np.indices((size,) * ndim) - r
    return (grid**2).sum(axis=0) <= r**2


def _diagonal_chain(pixels) -> np.ndarray:
    a = np.zeros((4, 5), np.float32)
    for r, c in pixels:
        a[r, c] = 1
    return a


def _inside_convex(poly: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Pixels strictly left of every edge of a counter-clockwise convex polygon."""
    y, x = np.mgrid[:ny, :nx]
    inside = np.ones((ny, nx), bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        inside &= (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0
    return inside


def test_rss_removes_exactly_the_reduced_axes():
    x = torch.randn(4, 3, 5, dtype=torch.complex64)
    ref = x.abs().square().sum(1).sqrt()
    torch.testing.assert_close(bartorch.rss(x, 1), ref.to(torch.complex64), rtol=1e-5, atol=1e-5)
    assert bartorch.rss(x, 0).shape == (3, 5)
    assert bartorch.rss(x, (0, 2)).shape == (3,)
    assert bartorch.rss(torch.randn(1, 4, 5, dtype=torch.complex64), 1).shape == (1, 5)
    assert bartorch.rss(x, 0, keepdim=True).shape == (1, 3, 5)
    assert bartorch.rss(x, (0, 2), keepdim=True).shape == (1, 3, 1)
