"""Tests for bartorch.lib: encoding operators and CG solver.

All tests require the compiled _bartorch_ext C++ extension.
"""

from __future__ import annotations

import torch

import bartorch.lib as bl
import bartorch.tools as bt

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NC = 4
_NY = 64
_NX = 64


def _nrmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalised root-mean-square error."""
    err = (a - b).abs().pow(2).sum().sqrt()
    ref = b.abs().pow(2).sum().sqrt()
    return (err / ref).item()


def _make_cartesian_data():
    """Return (ksp, sens, img) for a small 2-D Cartesian problem."""
    img = bt.phantom([_NY, _NX])  # (1, ny, nx)
    ksp_calib = bt.phantom([_NY, _NX], kspace=True, ncoils=_NC)
    sens = bt.ecalib(ksp_calib, calib_size=16, maps=1)  # (1, nc, 1, ny, nx)
    coil_imgs = sens[0] * img  # (nc, 1, ny, nx)
    ksp = bt.fft(coil_imgs, axes=(-1, -2))  # (nc, 1, ny, nx)
    return ksp, sens, img


# ---------------------------------------------------------------------------
# encoding_op: construction and shapes
# ---------------------------------------------------------------------------


def test_encoding_op_returns_bartlinop():
    """encoding_op returns a BartLinop instance."""
    _, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)
    assert isinstance(E, bl.BartLinop)


def test_encoding_op_ishape_oshape():
    """encoding_op ishape matches image domain, oshape matches k-space."""
    ksp, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)
    # Domain (ishape): image space — spatial dims × maps, no coil.
    # BART maps Fortran (nx, ny, 1, nc, 1) → img Fortran (nx, ny, 1, 1, 1)
    # C-order ishape is obtained by reversing and trimming trailing 1s
    # (BART drops singleton dimensions at the end of the shape).
    assert E.ishape[-1] == _NX
    assert E.ishape[-2] == _NY
    # Codomain (oshape): k-space — spatial × coil.
    assert E.oshape[-1] == _NX
    assert E.oshape[-2] == _NY
    assert _NC in E.oshape


def test_encoding_op_repr():
    """BartLinop repr includes ishape and oshape."""
    _, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)
    r = repr(E)
    assert "BartLinop" in r
    assert "ishape" in r
    assert "oshape" in r


def test_encoding_op_with_explicit_ksp_shape():
    """encoding_op accepts an explicit ksp_shape argument."""
    ksp, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens, ksp_shape=ksp.shape)
    # Should succeed and produce consistent shapes.
    assert E.oshape[-1] == _NX


# ---------------------------------------------------------------------------
# encoding_op: forward / adjoint application
# ---------------------------------------------------------------------------


def test_forward_output_shape():
    """E.forward(img) produces output matching E.oshape."""
    _, sens, img = _make_cartesian_data()
    E = bl.encoding_op(sens)
    # Pad img to ishape if needed (BART may expect leading 1-dims).
    x = img.reshape(E.ishape)
    y = E.forward(x)
    assert y.shape == torch.Size(E.oshape)


def test_adjoint_output_shape():
    """E.adjoint(ksp) produces output matching E.ishape."""
    ksp, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)
    y = ksp.reshape(E.oshape)
    x = E.adjoint(y)
    assert x.shape == torch.Size(E.ishape)


def test_forward_output_dtype():
    """E.forward() output is complex64."""
    _, sens, img = _make_cartesian_data()
    E = bl.encoding_op(sens)
    x = img.reshape(E.ishape)
    y = E.forward(x)
    assert y.dtype == torch.complex64


def test_call_alias():
    """E(x) is an alias for E.forward(x)."""
    _, sens, img = _make_cartesian_data()
    E = bl.encoding_op(sens)
    x = img.reshape(E.ishape)
    y1 = E(x)
    y2 = E.forward(x)
    assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Adjointness test: <Ax, y> == <x, A^H y>
# ---------------------------------------------------------------------------


def test_adjointness():
    """Inner-product adjointness: <Ax, y> == <x, A^H y> up to 1e-3."""
    _, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)

    torch.manual_seed(0)
    x = torch.randn(*E.ishape, dtype=torch.complex64)
    y = torch.randn(*E.oshape, dtype=torch.complex64)

    Ax = E.forward(x)
    AHy = E.adjoint(y)

    lhs = (Ax.conj() * y).sum()
    rhs = (x.conj() * AHy).sum()

    scale = max(lhs.abs().item(), 1e-10)
    err = (lhs - rhs).abs().item() / scale
    assert err < 1e-3, f"adjointness error {err:.2e}"


# ---------------------------------------------------------------------------
# Normal operator: A^H A == A^H (A x)
# ---------------------------------------------------------------------------


def test_normal_matches_adjoint_of_forward():
    """E.normal(x) == E.adjoint(E.forward(x)) up to 1e-4."""
    _, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)

    torch.manual_seed(1)
    x = torch.randn(*E.ishape, dtype=torch.complex64)

    n1 = E.normal(x)
    n2 = E.adjoint(E.forward(x))

    err = _nrmse(n1, n2)
    assert err < 1e-4, f"normal vs adjoint(forward) nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# CG solver via BartLinop.solve
# ---------------------------------------------------------------------------


def test_solve_returns_correct_shape():
    """E.solve(y) returns tensor with shape E.ishape."""
    ksp, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)
    y = ksp.reshape(E.oshape)
    x = E.solve(y, maxiter=5)
    assert x.shape == torch.Size(E.ishape)


def test_solve_dtype():
    """E.solve() output is complex64."""
    ksp, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)
    y = ksp.reshape(E.oshape)
    x = E.solve(y, maxiter=5)
    assert x.dtype == torch.complex64


def test_conjgrad_solve_consistency():
    """conjgrad_solve(E, y) matches E.solve(y)."""
    ksp, sens, _ = _make_cartesian_data()
    E = bl.encoding_op(sens)
    y = ksp.reshape(E.oshape)
    x1 = E.solve(y, maxiter=10, lam=1e-3)
    x2 = bl.conjgrad_solve(E, y, maxiter=10, lam=1e-3)
    assert torch.allclose(x1, x2)


def test_solve_convergence():
    """CG reconstruction is closer to ground truth than the zero image."""
    _, sens, img = _make_cartesian_data()
    E = bl.encoding_op(sens)

    # Generate k-space using the BART SENSE forward model so that the data are
    # consistent with the operator (bt.fft uses a centred/modulated FFT that
    # differs from the uncentred FFT used internally by BART's maps_create).
    img_ref = img.reshape(E.ishape)
    y = E.forward(img_ref)

    x_rec = E.solve(y, maxiter=30, lam=1e-4)

    err_rec = _nrmse(x_rec.real, img_ref.real)
    err_zero = _nrmse(torch.zeros_like(x_rec.real), img_ref.real)

    assert err_rec < err_zero, (
        f"CG reconstruction is no better than zero: rec={err_rec:.4f} zero={err_zero:.4f}"
    )


# ---------------------------------------------------------------------------
# Non-Cartesian encoding operator
# ---------------------------------------------------------------------------


def test_noncartesian_encoding_op():
    """encoding_op with traj creates an operator and applies correctly."""
    traj = bt.traj(r=True, x=_NX, y=_NX)
    # BART non-Cartesian ksp shape: C-order (nc, nspokes, nsamples, 1).
    # The trailing 1 maps to BART Fortran dim 0 = 1 (the trivial spatial
    # x-dimension; BART requires ksp_dims[0]==1 for NUFFT k-space).
    ksp_shape = (_NC, _NX, _NX, 1)

    ksp_calib = bt.phantom([_NY, _NX], kspace=True, ncoils=_NC)
    sens = bt.ecalib(ksp_calib, calib_size=16, maps=1)

    E = bl.encoding_op(sens, ksp_shape=ksp_shape, traj=traj)
    assert isinstance(E, bl.BartLinop)
    assert _NX in E.oshape


def test_noncartesian_forward_shape():
    """Non-Cartesian E.forward output matches E.oshape."""
    traj = bt.traj(r=True, x=_NX, y=_NX)
    # BART non-Cartesian ksp shape: C-order (nc, nspokes, nsamples, 1).
    # The trailing 1 maps to BART Fortran dim 0 = 1 (required by NUFFT).
    ksp_shape = (_NC, _NX, _NX, 1)
    ksp_calib = bt.phantom([_NY, _NX], kspace=True, ncoils=_NC)
    sens = bt.ecalib(ksp_calib, calib_size=16, maps=1)

    E = bl.encoding_op(sens, ksp_shape=ksp_shape, traj=traj)
    torch.manual_seed(2)
    x = torch.randn(*E.ishape, dtype=torch.complex64)
    y = E.forward(x)
    assert y.shape == torch.Size(E.oshape)
