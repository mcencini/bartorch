"""The compiled core: tools run in-process on tensors, against references
that are not BART.
"""

import ctypes
import os

import numpy as np
import pytest
import torch

import bartorch
import bartorch._lib
import bartorch.tools as bt
from bartorch.core.graph import build_argv, dispatch


def test_library_reports_the_pinned_bart_version():
    assert bartorch.bart_version().startswith("v1.0")


def test_build_info_names_the_nested_function_mode():
    info = bartorch.build_info()
    assert "nested=clang-blocks" in info or "nested=gcc-heap-trampolines" in info


def test_argv_flags_positionals_inputs_output_in_that_order():
    argv = build_argv(
        "fft", ["a.mem"], "o.mem", [3], {"u": True, "i": False, "x": (1, 2), "long_name": 4}
    )
    assert argv == ["fft", "-u", "-x", "1:2", "--long-name", "4", "3", "a.mem", "o.mem"]


def test_list_flags_repeat():
    argv = build_argv("pics", [], None, [], {"R": ["W:7:0:0.01", "T:7:0:0.02"]})
    assert argv == ["pics", "-R", "W:7:0:0.01", "-R", "T:7:0:0.02"]


def test_fft_matches_numpy_on_the_last_axis():
    x = torch.randn(4, 32, dtype=torch.complex64)
    y = bt.fft(x, axes=-1)
    ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x.numpy(), axes=-1), axis=-1), axes=-1)
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_fft_matches_numpy_on_two_axes():
    x = torch.randn(3, 16, 24, dtype=torch.complex64)
    y = bt.fft(x, axes=(-1, -2))
    ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x.numpy(), axes=(-1, -2))), axes=(-1, -2))
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_inverse_fft_round_trips():
    x = torch.randn(8, 8, dtype=torch.complex64)
    y = bt.ifft(bt.fft(x, axes=(-1, -2), unitary=True), axes=(-1, -2), unitary=True)
    torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-4)


def test_output_is_a_tensor_that_was_never_copied():
    x = torch.randn(8, 8, dtype=torch.complex64)
    y = bt.fft(x, axes=-1)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.complex64
    assert y.is_contiguous()


def test_phantom_shape_follows_c_order():
    p = bt.phantom([64, 64])
    assert p.shape[-2:] == (64, 64)


def test_scalar_tool_returns_its_text():
    x = torch.ones(4, dtype=torch.complex64)
    text = dispatch("nrmse", [x, x], False)
    assert float(text) == pytest.approx(0.0)


def test_a_failing_tool_raises_with_barts_message():
    x = torch.ones(4, dtype=torch.complex64)
    with pytest.raises(bartorch.BartError):
        dispatch("fft", [x], None, _pos=["notanumber"])


def test_registry_is_empty_after_a_failure():
    from bartorch._lib import library

    x = torch.ones(4, dtype=torch.complex64)
    with pytest.raises(bartorch.BartError):
        dispatch("fft", [x], None, _pos=["notanumber"])
    assert library().bartorch_unlink_all() == 0


def test_rss_matches_numpy():
    x = torch.randn(8, 16, 16, dtype=torch.complex64)
    y = bt.rss(x, axes=0)
    ref = np.sqrt((np.abs(x.numpy()) ** 2).sum(0))
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_svd_through_the_lapack_backend():
    a = torch.randn(6, 4, dtype=torch.complex64)
    a0 = a.clone()
    u, s, vh = bt.svd(a)
    # BART sees the C-order (6, 4) tensor as the Fortran matrix A^T (4 x 6),
    # so in C order the factors read A = vh[:, :4] @ diag(s) @ u.
    np.testing.assert_allclose(
        np.sort(s.real.numpy())[::-1], np.linalg.svd(a0.numpy(), compute_uv=False), rtol=1e-4
    )
    recon = vh.numpy()[:, :4] @ np.diag(s.numpy().ravel()) @ u.numpy()
    np.testing.assert_allclose(recon, a0.numpy(), rtol=1e-3, atol=1e-4)


def test_inputs_are_left_untouched_by_a_tool_that_writes_into_them():
    a = torch.randn(6, 4, dtype=torch.complex64)
    a0 = a.clone()
    bt.svd(a)
    torch.testing.assert_close(a, a0)


def test_an_array_passed_as_a_flag_reaches_the_tool_and_is_left_untouched():
    # `pics -t` takes a trajectory, `-p` a pattern, `-B` a basis: a flag's
    # value can be an array, registered like any other input.
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    before = traj.clone()
    image = bt.phantom([n, n]).reshape(1, n, n)
    kspace = bt.nufft(traj, image)
    maps = torch.ones(1, n, n, dtype=torch.complex64)

    recon = bt.pics(kspace, maps, t=traj)

    assert recon.shape == (n, n)
    torch.testing.assert_close(traj, before)


def test_scratch_inputs_skip_the_copy():
    a = torch.randn(6, 4, dtype=torch.complex64)
    a0 = a.clone()
    bartorch.set_copy_inputs(False)
    try:
        bt.svd(a)
    finally:
        bartorch.set_copy_inputs(True)
    assert not torch.allclose(a, a0)


def test_every_blas_and_lapack_routine_comes_from_a_compiled_library():
    sources = bartorch.backend_sources()
    assert sources, "the backend table was never filled"
    missing = [name for name, src in sources.items() if src == "missing"]
    assert not missing, f"no compiled routine found for {missing}"


def test_lapack_is_served_by_a_library_not_by_the_built_in_reference():
    # The library carries reference BLAS for the handful of level-1 and level-2
    # routines, but no LAPACK at all: every LAPACK entry must resolve to a
    # library in the process.
    lapack = {
        name: src
        for name, src in bartorch.backend_sources().items()
        if name[1:].startswith(
            (
                "heev",
                "hegv",
                "gesdd",
                "gesvd",
                "geqrf",
                "ungqr",
                "potrf",
                "trtri",
                "trtrs",
                "getrf",
                "getri",
                "gees",
                "trsyl",
                "gesv",
            )
        )
    }
    assert lapack
    assert all(src not in ("reference", "missing") for src in lapack.values()), lapack


def test_an_installed_mkl_serves_every_routine():
    # MKL is preferred when the mkl extra is installed, and it has to cover
    # everything rather than leave gaps for another library to fill.
    from bartorch import _backend

    if os.environ.get("BARTORCH_BLAS_LIBRARY"):
        pytest.skip("a backend was asked for explicitly")
    if _backend._mkl_library() is None:
        pytest.skip("no MKL installed in this environment")
    assert set(bartorch.backend_sources().values()) == {"mkl"}


def test_a_named_backend_can_be_asked_for():
    from bartorch import _backend

    providers = {p.name for p in _backend._providers()}
    assert "scipy" in providers, "SciPy is a dependency and must always be a candidate"


def test_a_failed_assertion_inside_bart_reaches_the_caller():
    """BART checks its arguments with assert, and assert must not end the process.

    Coil images and k-space that disagree on their dimensions trip an
    assertion deep inside BART's own NUFFT.  glibc's assert would abort, so
    the library answers __assert_fail itself and puts it back on BART's error
    path, where the error catcher turns it into a return code.
    """
    import bartorch.tools as bt
    from bartorch import linop

    n = 16
    traj = bt.traj(x=n, y=8, r=True)
    with pytest.raises(bartorch.BartError):
        linop.NUFFT(traj, (8, 1, n, n), toeplitz=False)
