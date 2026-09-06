"""Tests for the torch_prior plug-and-play path in bartorch.tools.pics.

These tests cover:

* ``_bart_img_dims_from_kspace`` — converts a kspace tensor shape to the
  BART Fortran-order ``img_dims`` array (length 16).
* ``pics(..., torch_prior=fn)`` argument validation via signature inspection.
"""

from __future__ import annotations

import inspect

import torch

from bartorch.tools._commands import _bart_img_dims_from_kspace, pics

__all__: list[str] = []

_BART_DIMS = 16
_COIL_DIM = 3


# ---------------------------------------------------------------------------
# _bart_img_dims_from_kspace
# ---------------------------------------------------------------------------


def test_img_dims_2d_4d_tensor():
    """2-D case: (nc, 1, ny, nx) → img_dims with nx,ny spatial, coil=1."""
    nc, ny, nx = 8, 64, 96
    ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)

    assert len(dims) == _BART_DIMS
    # Fortran reversed from C-order [nc, 1, ny, nx] → [nx, ny, 1, nc, ...]
    assert dims[0] == nx
    assert dims[1] == ny
    assert dims[2] == 1
    # Coil dim (Fortran index 3) must be zeroed to 1 for img_dims
    assert dims[_COIL_DIM] == 1
    # All remaining dims are 1
    assert all(d == 1 for d in dims[4:])


def test_img_dims_3d_tensor():
    """3-D case: (nc, nz, ny, nx) → img_dims with full spatial, coil=1."""
    nc, nz, ny, nx = 4, 16, 32, 48
    ksp = torch.zeros(nc, nz, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)

    assert len(dims) == _BART_DIMS
    assert dims[0] == nx
    assert dims[1] == ny
    assert dims[2] == nz
    assert dims[_COIL_DIM] == 1  # coil zeroed
    assert all(d == 1 for d in dims[4:])


def test_img_dims_returns_list_of_ints():
    ksp = torch.zeros(2, 1, 8, 8, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    assert isinstance(dims, list)
    assert all(isinstance(d, int) for d in dims)


def test_img_dims_length_always_16():
    for shape in [(2, 1, 4, 4), (4, 8, 16, 16), (1, 1, 32, 32)]:
        ksp = torch.zeros(*shape, dtype=torch.complex64)
        assert len(_bart_img_dims_from_kspace(ksp)) == _BART_DIMS


def test_img_dims_coil_dim_always_1():
    """Coil dim (Fortran index 3) must be 1 regardless of nc."""
    for nc in (1, 4, 16, 32):
        ksp = torch.zeros(nc, 1, 8, 8, dtype=torch.complex64)
        dims = _bart_img_dims_from_kspace(ksp)
        assert dims[_COIL_DIM] == 1, f"coil dim not 1 for nc={nc}: {dims}"


def test_img_dims_single_coil():
    nc, ny, nx = 1, 16, 16
    ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    assert dims[0] == nx
    assert dims[1] == ny
    assert dims[_COIL_DIM] == 1


def test_img_dims_non_square_spatial():
    nc, ny, nx = 2, 32, 64
    ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    assert dims[0] == nx
    assert dims[1] == ny


def test_img_dims_3d_coil_zeroed():
    """3-D kspace: original nc must NOT appear at Fortran dim 3 in img_dims."""
    nc, nz, ny, nx = 6, 10, 20, 30
    ksp = torch.zeros(nc, nz, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    # Before zeroing, ksp_dims[3] == nc (6). After zeroing it must be 1.
    assert dims[_COIL_DIM] == 1
    # The original nc value must not appear at the coil position.
    assert dims[_COIL_DIM] != nc


# ---------------------------------------------------------------------------
# pics() — torch_prior argument validation (signature inspection only)
# ---------------------------------------------------------------------------


def test_torch_prior_lambda_default_is_float():
    """torch_prior_lambda has a sane numeric default (1.0)."""
    sig = inspect.signature(pics)
    default = sig.parameters["torch_prior_lambda"].default
    assert isinstance(default, float)
    assert default == 1.0


def test_pics_has_torch_prior_parameter():
    """pics() signature must include torch_prior keyword."""
    sig = inspect.signature(pics)
    assert "torch_prior" in sig.parameters


def test_pics_has_torch_prior_lambda_parameter():
    """pics() signature must include torch_prior_lambda keyword."""
    sig = inspect.signature(pics)
    assert "torch_prior_lambda" in sig.parameters
