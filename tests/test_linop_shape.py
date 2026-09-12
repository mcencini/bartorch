"""The operators that rearrange, reduce or restrict a shape.

Each one is a BART constructor, and BART counts its dimensions the other way
round from torch, so what is checked here is mostly the turning-around: that
the axis the caller named is the axis BART moved.  Every claim is made against
the torch operation of the same name, which is the thing a reader will expect
the operator to agree with.
"""

import pytest
import torch

from bartorch import linop


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _adjointness(A, seed=0):
    """The dot test, as a relative error; meaningful only for a complex-linear A."""
    torch.manual_seed(seed)
    x = _rand(*A.ishape)
    y = _rand(*A.oshape)
    left = torch.vdot(A(x).flatten(), y.flatten())
    right = torch.vdot(x.flatten(), A.H(y).flatten())
    return (left - right).abs().item() / max(left.abs().item(), 1e-12)


SHAPE = (3, 4, 5)


# --- each against the torch operation it is named for -----------------------


def test_flip_reverses_the_axes_it_was_given():
    x = _rand(*SHAPE)
    torch.testing.assert_close(linop.Flip(SHAPE, axes=-1)(x), x.flip(-1))
    torch.testing.assert_close(linop.Flip(SHAPE, axes=0)(x), x.flip(0))
    torch.testing.assert_close(linop.Flip(SHAPE, axes=(0, -1))(x), x.flip(0).flip(-1))


def test_sum_keeps_the_axis_it_summed():
    """BART leaves the summed axis at one rather than dropping it."""
    x = _rand(*SHAPE)
    S = linop.Sum(SHAPE, axes=0)
    assert S.oshape == (1, 4, 5)
    torch.testing.assert_close(S(x), x.sum(0, keepdim=True), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        linop.Sum(SHAPE, axes=(0, 2))(x), x.sum((0, 2), keepdim=True), rtol=1e-5, atol=1e-5
    )


def test_mean_divides_by_how_many_it_summed():
    x = _rand(*SHAPE)
    torch.testing.assert_close(
        linop.Mean(SHAPE, axes=1)(x), x.mean(1, keepdim=True), rtol=1e-5, atol=1e-5
    )


def test_repeat_is_the_adjoint_of_sum():
    x = _rand(1, 4, 5)
    R = linop.Repeat(SHAPE, axes=0)
    assert R.ishape == (1, 4, 5) and R.oshape == SHAPE
    torch.testing.assert_close(R(x), x.expand(SHAPE).contiguous())
    y = _rand(*SHAPE)
    torch.testing.assert_close(R.adjoint(y), y.sum(0, keepdim=True), rtol=1e-5, atol=1e-5)


def test_reshape_reads_the_same_elements():
    x = _rand(*SHAPE)
    torch.testing.assert_close(linop.Reshape((12, 5), SHAPE)(x), x.reshape(12, 5))


def test_reshape_refuses_a_different_number_of_elements():
    with pytest.raises(ValueError, match="elements against"):
        linop.Reshape((7, 5), SHAPE)


def test_transpose_exchanges_two_axes():
    x = _rand(*SHAPE)
    T = linop.Transpose(SHAPE, 0, 2)
    assert T.oshape == (5, 4, 3)
    torch.testing.assert_close(T(x), x.transpose(0, 2).contiguous())


def test_permute_reads_the_way_torchs_does():
    """Axis i of the result is axis order[i] of the input, in both."""
    x = _rand(*SHAPE)
    for order in ((2, 0, 1), (1, 2, 0), (0, 2, 1), (2, 1, 0)):
        P = linop.Permute(SHAPE, order)
        assert P.oshape == tuple(SHAPE[o] for o in order)
        torch.testing.assert_close(P(x), x.permute(order).contiguous())


def test_permute_refuses_something_that_is_not_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        linop.Permute(SHAPE, (0, 0, 1))


def test_roll_wraps_round_like_torchs():
    x = _rand(*SHAPE)
    torch.testing.assert_close(linop.Roll(SHAPE, 2, axis=-1)(x), x.roll(2, dims=-1))
    torch.testing.assert_close(linop.Roll(SHAPE, -1, axis=0)(x), x.roll(-1, dims=0))


def test_extract_takes_the_block_it_was_given():
    x = _rand(*SHAPE)
    E = linop.Extract((1, 0, 2), (2, 4, 3), SHAPE)
    torch.testing.assert_close(E(x), x[1:3, 0:4, 2:5])


def test_extract_puts_the_block_back_and_leaves_the_rest_zero():
    E = linop.Extract((1, 0, 2), (2, 4, 3), SHAPE)
    block = _rand(2, 4, 3)
    back = E.adjoint(block)
    torch.testing.assert_close(back[1:3, 0:4, 2:5], block)
    outside = back.clone()
    outside[1:3, 0:4, 2:5] = 0
    assert outside.abs().max() == 0, "the adjoint wrote outside the block"


def test_extract_refuses_a_block_that_does_not_fit():
    with pytest.raises(ValueError, match="does not fit"):
        linop.Extract((2, 0, 0), (2, 4, 5), SHAPE)


def test_pad_adds_zeros_at_both_ends():
    x = _rand(*SHAPE)
    P = linop.Pad(SHAPE, (0, 0, 1), (0, 0, 2))
    assert P.oshape == (3, 4, 8)
    torch.testing.assert_close(P(x), torch.nn.functional.pad(x, (1, 2)))


def test_pad_takes_one_number_for_every_axis():
    P = linop.Pad(SHAPE, 1)
    assert P.oshape == (5, 6, 7)


def test_pad_refuses_to_crop():
    with pytest.raises(ValueError, match="cannot be negative"):
        linop.Pad(SHAPE, (-1, 0, 0))


def test_resize_crops_and_fills_about_the_centre():
    """What `resize -c` does, which is what an FFT's conventions want."""
    x = _rand(1, 1, 4)
    grown = linop.Resize((1, 1, 8), (1, 1, 4))(x)
    assert grown.shape == (1, 1, 8)
    torch.testing.assert_close(grown[0, 0, 2:6], x[0, 0])
    assert grown[0, 0, :2].abs().sum() == 0 and grown[0, 0, 6:].abs().sum() == 0
    torch.testing.assert_close(linop.Resize((1, 1, 4), (1, 1, 8))(grown), x)


def test_real_takes_the_real_part():
    x = _rand(*SHAPE)
    torch.testing.assert_close(linop.Real(SHAPE)(x), x.real.to(torch.complex64))


# --- Hankel ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "axis", "window"),
    [((64,), 0, 16), ((64, 8), 0, 16), ((8, 64), 1, 4), ((6,), 0, 6), ((5, 7, 3), 1, 2)],
)
def test_a_sliding_window_is_torchs_unfold(shape, axis, window):
    """Same windows, same order, window last -- which is where unfold puts it."""
    x = _rand(*shape)
    H = linop.Hankel(shape, axis=axis, window=window)
    torch.testing.assert_close(H(x), x.unfold(axis, window, 1))


def test_the_window_shortens_the_axis_it_slides_along():
    H = linop.Hankel((64, 8), axis=0, window=16)
    assert H.ishape == (64, 8)
    assert H.oshape == (49, 8, 16)


def test_a_window_the_length_of_the_axis_gives_one_position():
    H = linop.Hankel((6,), axis=0, window=6)
    assert H.oshape == (1, 6)


def test_the_adjoint_adds_each_sample_back_into_every_window_it_was_in():
    """Which is what makes this an operator rather than a view."""
    H = linop.Hankel((8,), axis=0, window=3)
    ones = torch.ones(*H.oshape, dtype=torch.complex64)
    # sample i appears in as many windows as overlap it: 1, 2, 3, 3, 3, 3, 2, 1
    torch.testing.assert_close(
        H.adjoint(ones).real,
        torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 1.0]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_a_window_that_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        linop.Hankel((8,), axis=0, window=9)


def test_an_empty_window_is_refused():
    with pytest.raises(ValueError, match="nothing in it"):
        linop.Hankel((8,), axis=0, window=0)


# --- what the shapes and the adjoints have to hold ---------------------------


def _complex_linear_operators():
    return [
        linop.Flip(SHAPE, axes=-1),
        linop.Sum(SHAPE, axes=0),
        linop.ScaledSum(SHAPE, axes=(0, 2)),
        linop.Mean(SHAPE, axes=(0, 1)),
        linop.Repeat(SHAPE, axes=2),
        linop.Reshape((12, 5), SHAPE),
        linop.Transpose(SHAPE, 0, 2),
        linop.Permute(SHAPE, (2, 0, 1)),
        linop.Roll(SHAPE, 2, axis=-1),
        linop.Extract((1, 0, 2), (2, 4, 3), SHAPE),
        linop.Pad(SHAPE, (0, 1, 1)),
        linop.Resize((3, 2, 7), SHAPE),
        linop.Hankel(SHAPE, axis=1, window=2),
    ]


@pytest.mark.parametrize("A", _complex_linear_operators(), ids=lambda A: type(A).__name__)
def test_each_ones_adjoint_is_its_adjoint(A):
    assert _adjointness(A) < 1e-4


@pytest.mark.parametrize("A", _complex_linear_operators(), ids=lambda A: type(A).__name__)
def test_each_is_one_bart_operator(A):
    assert A._native
    assert hasattr(A, "_h")


@pytest.mark.parametrize("A", _complex_linear_operators(), ids=lambda A: type(A).__name__)
def test_each_composes_with_the_rest_of_the_algebra(A):
    """Which is the point of them being operators rather than functions."""
    combined = 2.0 * A.H @ A
    x = _rand(*A.ishape)
    torch.testing.assert_close(combined(x), 2.0 * A.adjoint(A(x)), rtol=1e-4, atol=1e-4)


def test_real_is_linear_over_the_reals_only():
    """So it is left out of the dot test above rather than quietly passing it."""
    R = linop.Real(SHAPE)
    assert _adjointness(R) > 1e-3, "Real is complex-linear after all, which it should not be"
    x = torch.randn(*SHAPE).to(torch.complex64)
    y = torch.randn(*SHAPE).to(torch.complex64)
    left = torch.vdot(R(x).flatten(), y.flatten())
    right = torch.vdot(x.flatten(), R.adjoint(y).flatten())
    assert (left - right).abs().item() < 1e-4, "Real is not self-adjoint on real vectors"


# --- the closed-form pseudo-inverse, and the one BART gets wrong -------------


def test_only_the_scaled_sum_takes_barts_closed_form():
    """`linops/sum.c` is the only place in BART that supplies a norm_inv.

    It attaches the same routine to the plain sum and to the scaled one, and
    the routine divides by a count that ``linop_sum_create`` overwrites with
    one after the shared data is built.  So it answers for the scaled operator
    in both cases, and only the scaled operator may use it.
    """
    from bartorch._lib import library

    claims = {}
    for name, A in (
        ("Sum", linop.Sum(SHAPE, axes=0)),
        ("ScaledSum", linop.ScaledSum(SHAPE, axes=0)),
        ("Mean", linop.Mean(SHAPE, axes=0)),
    ):
        held = A._bart()
        claims[name] = (bool(library().bartorch_linop_has_pseudo_inv(held._h.ptr)), A._exact_pinv)

    assert claims["ScaledSum"] == (True, True)
    assert claims["Sum"] == (True, False), "BART offers one for the plain sum; we must not take it"
    assert claims["Mean"] == (False, False)


def test_the_scaled_sum_is_the_sum_over_the_root_of_the_count():
    x = _rand(*SHAPE)
    torch.testing.assert_close(
        linop.ScaledSum(SHAPE, axes=0)(x),
        x.sum(0, keepdim=True) / SHAPE[0] ** 0.5,
        rtol=1e-5,
        atol=1e-5,
    )


def test_its_normal_is_a_projection():
    """Which is what makes the closed form a closed form."""
    S = linop.ScaledSum(SHAPE, axes=0)
    x = _rand(*SHAPE)
    once = S.gram()(x)
    torch.testing.assert_close(S.gram()(once), once, rtol=1e-4, atol=1e-4)
    assert abs(S.opnorm() - 1.0) < 1e-3


@pytest.mark.parametrize("damp", [0.25, 1.0, 3.0])
def test_the_closed_form_solves_the_damped_normal_equations(damp):
    S = linop.ScaledSum(SHAPE, axes=0)
    y = _rand(1, 4, 5)
    x = S.pinv(y, damp=damp)
    torch.testing.assert_close(S.adjoint(S(x)) + damp * x, S.adjoint(y), rtol=1e-4, atol=1e-4)


def test_the_closed_form_agrees_with_the_solver_it_replaces():
    from bartorch.optim import CG

    S = linop.ScaledSum(SHAPE, axes=0)
    y = _rand(1, 4, 5)
    torch.testing.assert_close(
        S.pinv(y, damp=0.5), CG(0.5, maxiter=300, tol=1e-12)(y, S, None), rtol=1e-3, atol=1e-3
    )


def test_the_plain_sum_goes_through_the_solver_and_gets_the_right_answer():
    """The answer BART's closed form would have given here is not this one."""
    S = linop.Sum(SHAPE, axes=0)
    y = _rand(1, 4, 5)
    x = S.pinv(y, damp=0.5, maxiter=300, tol=1e-12)
    torch.testing.assert_close(S.adjoint(S(x)) + 0.5 * x, S.adjoint(y), rtol=1e-3, atol=1e-3)


def test_a_solver_keyword_is_refused_where_there_is_no_solver():
    S = linop.ScaledSum(SHAPE, axes=0)
    with pytest.raises(TypeError, match="no solver to configure"):
        S.pinv(_rand(1, 4, 5), maxiter=10)
