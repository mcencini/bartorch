"""Filtering, differencing and multiplying by a matrix.

Each is one of BART's constructors, checked against what it claims to be
rather than against itself: a convolution against an explicit circular one, a
difference against the difference, a matrix against an einsum.
"""

import numpy as np
import pytest
import torch

from bartorch import linop


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _adjointness(A, seed=0):
    torch.manual_seed(seed)
    x, y = _rand(*A.ishape), _rand(*A.oshape)
    left = torch.vdot(A(x).flatten(), y.flatten())
    right = torch.vdot(x.flatten(), A.H(y).flatten())
    return (left - right).abs().item() / max(left.abs().item(), 1e-12)


# --- Matrix ------------------------------------------------------------------


def test_a_matrix_contracts_the_axis_the_shapes_differ_along():
    t, k, n = 6, 3, 5
    matrix = _rand(k, t, 1)
    A = linop.Matrix(matrix, (k, 1, n), (1, t, n))
    x = _rand(1, t, n)
    want = torch.einsum("kt,tn->kn", matrix[:, :, 0], x[0])
    torch.testing.assert_close(A(x)[:, 0], want, rtol=1e-4, atol=1e-4)


def test_a_matrix_broadcasts_over_the_axes_it_has_one_of():
    """One operator, a different matrix per slice, is the same constructor."""
    t, k, n = 4, 2, 3
    per_slice = _rand(k, t, n)
    A = linop.Matrix(per_slice, (k, 1, n), (1, t, n))
    x = _rand(1, t, n)
    want = torch.einsum("ktn,tn->kn", per_slice, x[0])
    torch.testing.assert_close(A(x)[:, 0], want, rtol=1e-4, atol=1e-4)


def test_a_matrix_needs_one_axis_per_axis_of_the_operator():
    with pytest.raises(ValueError, match="one axis per axis"):
        linop.Matrix(_rand(3, 4), (3, 1, 5), (1, 4, 5))


def test_a_matrix_keeps_the_rank_it_has():
    with pytest.raises(ValueError, match="differ in rank"):
        linop.Matrix(_rand(3, 4, 1), (3, 5), (1, 4, 5))


# --- Convolve ----------------------------------------------------------------


@pytest.mark.parametrize(("mode", "size"), [("same", 8), ("wrap", 8), ("valid", 6), ("full", 10)])
def test_each_convolution_mode_gives_the_size_it_should(mode, size):
    A = linop.Convolve(_rand(3), (8,), axes=0, mode=mode)
    assert A.oshape == (size,)


def test_a_circular_convolution_is_one():
    """Against a multiplication in the Fourier domain, up to where BART centres it."""
    kernel, x = _rand(3), _rand(8)
    A = linop.Convolve(kernel, (8,), axes=0, mode="wrap")
    ref = np.fft.ifft(np.fft.fft(x.numpy()) * np.fft.fft(np.pad(kernel.numpy(), (0, 5))))
    np.testing.assert_allclose(
        np.abs(np.fft.fft(A(x).numpy())), np.abs(np.fft.fft(ref)), rtol=1e-3, atol=1e-3
    )


def test_valid_and_full_are_planned_as_causal_without_being_asked():
    """BART refuses them otherwise, and failing inside its planner is no answer."""
    for mode in ("valid", "full"):
        assert linop.Convolve(_rand(3), (8,), axes=0, mode=mode).direction == "causal"
    for mode in ("same", "wrap"):
        assert linop.Convolve(_rand(3), (8,), axes=0, mode=mode).direction == "symmetric"


def test_asking_for_a_direction_bart_will_not_plan_says_so_here():
    with pytest.raises(ValueError, match="only as a causal one"):
        linop.Convolve(_rand(3), (8,), axes=0, mode="valid", direction="symmetric")


def test_a_kernel_that_leaves_nothing_is_refused():
    with pytest.raises(ValueError, match="leaves nothing"):
        linop.Convolve(_rand(9), (8,), axes=0, mode="valid")


def test_an_unknown_mode_says_what_the_modes_are():
    with pytest.raises(ValueError, match="is not one of"):
        linop.Convolve(_rand(3), (8,), axes=0, mode="reflect")


# --- Gradient ----------------------------------------------------------------


def _ramp():
    return torch.arange(20, dtype=torch.float32).reshape(4, 5).to(torch.complex64)


def test_the_gradient_is_the_forward_difference_with_a_circular_end():
    z = _ramp()
    g = linop.Gradient((4, 5), axes=0)(z)
    assert g.shape == (1, 4, 5)
    # down a column of the ramp the step is five, and the last wraps to the first
    torch.testing.assert_close(
        g[0][:, 0].real, torch.tensor([5.0, 5.0, 5.0, -15.0]), rtol=1e-4, atol=1e-4
    )


def test_the_components_come_out_in_ascending_axis_order():
    """BART stacks them by bit position, which runs the other way to C axes."""
    z = _ramp()
    g = linop.Gradient((4, 5), axes=(0, 1))(z)
    assert g.shape == (2, 4, 5)
    torch.testing.assert_close(
        g[0][:, 0].real, torch.tensor([5.0, 5.0, 5.0, -15.0]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        g[1][0].real, torch.tensor([1.0, 1.0, 1.0, 1.0, -4.0]), rtol=1e-4, atol=1e-4
    )


def test_the_order_of_the_axes_it_was_given_does_not_matter():
    z = _ramp()
    torch.testing.assert_close(
        linop.Gradient((4, 5), axes=(1, 0))(z), linop.Gradient((4, 5), axes=(0, 1))(z)
    )


def test_a_constant_is_in_the_null_space():
    """Which is what a circular difference means, and what solving with one has to reckon with."""
    A = linop.Gradient((4, 5), axes=(0, 1))
    flat = torch.ones(4, 5, dtype=torch.complex64)
    assert A(flat).abs().max() < 1e-5


def test_naming_an_axis_twice_is_refused():
    with pytest.raises(ValueError, match="names an axis twice"):
        linop.Gradient((4, 5), axes=(0, 0))


# --- ComponentDiagonal -------------------------------------------------------


def test_it_scales_the_two_components_separately():
    diag = (torch.rand(5) + 1j * torch.rand(5)).to(torch.complex64)
    x = _rand(5)
    got = linop.ComponentDiagonal(diag, (5,))(x)
    want = torch.complex(x.real * diag.real, x.imag * diag.imag)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_a_real_diagonal_through_it_annihilates_the_imaginary_part():
    """Which is why it is not called RealDiagonal."""
    w = torch.rand(5).to(torch.complex64)
    x = _rand(5)
    got = linop.ComponentDiagonal(w, (5,))(x)
    assert got.imag.abs().max() < 1e-6
    torch.testing.assert_close(got.real, x.real * w.real, rtol=1e-5, atol=1e-5)


def test_the_real_diagonal_operator_is_the_ordinary_diagonal():
    """What a reader reaching for RealDiagonal actually wants."""
    w = torch.rand(5).to(torch.complex64)
    x, y = _rand(5), _rand(5)
    D = linop.Diagonal(w, (5,))
    torch.testing.assert_close(D(x), x * w, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(D.adjoint(y), y * w, rtol=1e-5, atol=1e-5)
    assert _adjointness(D) < 1e-4


def test_it_is_linear_over_the_reals_only():
    diag = (torch.rand(5) + 1j * torch.rand(5)).to(torch.complex64)
    A = linop.ComponentDiagonal(diag, (5,))
    assert _adjointness(A) > 1e-3, "it is complex-linear after all, which it should not be"


# --- what they all have to hold ----------------------------------------------


def _complex_linear():
    return [
        linop.Matrix(_rand(3, 6, 1), (3, 1, 5), (1, 6, 5)),
        linop.Convolve(_rand(3), (8,), axes=0, mode="same"),
        linop.Convolve(_rand(3), (8,), axes=0, mode="wrap"),
        linop.Convolve(_rand(3), (8,), axes=0, mode="valid"),
        linop.Convolve(_rand(3), (8,), axes=0, mode="full"),
        linop.Gradient((4, 5), axes=0),
        linop.Gradient((4, 5), axes=(0, 1)),
    ]


@pytest.mark.parametrize("A", _complex_linear(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_ones_adjoint_is_its_adjoint(A):
    assert _adjointness(A) < 1e-4


@pytest.mark.parametrize("A", _complex_linear(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_stays_one_bart_operator(A):
    assert A._native
    assert hasattr(A._bart(), "_h")


@pytest.mark.parametrize("A", _complex_linear(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_carries_on_into_the_algebra(A):
    x = _rand(*A.ishape)
    torch.testing.assert_close((2.0 * A.H @ A)(x), 2.0 * A.adjoint(A(x)), rtol=1e-4, atol=1e-4)


def test_a_total_variation_normal_operator_is_the_thing_this_is_for():
    """A^H A of the gradient is the graph Laplacian, which a solver will want."""
    A = linop.Gradient((8, 8), axes=(0, 1))
    x = _rand(8, 8)
    torch.testing.assert_close(A.gram()(x), A.adjoint(A(x)), rtol=1e-4, atol=1e-4)
