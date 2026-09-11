"""Tests for bartorch.io (NumPy CFL read/write)."""

from __future__ import annotations

import numpy as np
import pytest

from bartorch.io import readcfl, writecfl

__all__: list[str] = []


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def test_roundtrip_2d(tmp):
    arr = (np.random.randn(8, 16) + 1j * np.random.randn(8, 16)).astype(np.complex64)
    base = str(tmp / "test")
    writecfl(base, arr)
    arr2 = readcfl(base)
    np.testing.assert_array_almost_equal(arr, arr2)


def test_roundtrip_3d(tmp):
    arr = (np.random.randn(4, 8, 16) + 1j * np.random.randn(4, 8, 16)).astype(np.complex64)
    base = str(tmp / "test3d")
    writecfl(base, arr)
    arr2 = readcfl(base)
    assert arr2.shape == (4, 8, 16)
    np.testing.assert_array_almost_equal(arr, arr2)


def test_trailing_ones_stripped(tmp):
    arr = np.zeros((4, 1, 1), dtype=np.complex64)
    base = str(tmp / "ones")
    writecfl(base, arr)
    arr2 = readcfl(base)
    assert arr2.shape == (4,) or arr2.shape == (4, 1, 1)


def test_float_cast(tmp):
    arr = np.ones((4, 4), dtype=np.float32)
    base = str(tmp / "float")
    writecfl(base, arr)
    arr2 = readcfl(base)
    assert arr2.dtype == np.complex64
