"""Putting operators side by side, and slicing one.

The only functions in ``bartorch.linop``: they take operators and give back an
operator, which is the one thing that cannot be spelled as arithmetic or as
indexing.  Each is checked against the torch operation of the same name
applied to what the operators return, which is what the function means.
"""

import pytest
import torch

from bartorch import linop

SHAPE = (4, 6)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _adjointness(A, seed=0):
    torch.manual_seed(seed)
    x, y = _rand(*A.ishape), _rand(*A.oshape)
    left = torch.vdot(A(x).flatten(), y.flatten())
    right = torch.vdot(x.flatten(), A.H(y).flatten())
    return (left - right).abs().item() / max(left.abs().item(), 1e-12)


@pytest.fixture
def pair():
    return linop.FFT(SHAPE, axes=-1), linop.Identity(SHAPE)


# --- what each one means -----------------------------------------------------


def test_concatenate_lays_the_results_end_to_end(pair):
    F, Id = pair
    x = _rand(*SHAPE)
    A = linop.concatenate([F, Id])
    assert A.oshape == (8, 6) and A.ishape == SHAPE
    torch.testing.assert_close(A(x), torch.cat([F(x), Id(x)], 0), rtol=1e-4, atol=1e-4)


def test_concatenate_along_another_axis(pair):
    F, Id = pair
    x = _rand(*SHAPE)
    A = linop.concatenate([F, Id], axis=1)
    assert A.oshape == (4, 12)
    torch.testing.assert_close(A(x), torch.cat([F(x), Id(x)], 1), rtol=1e-4, atol=1e-4)


def test_stack_puts_the_results_on_a_new_axis(pair):
    F, Id = pair
    x = _rand(*SHAPE)
    A = linop.stack([F, Id])
    assert A.oshape == (2, 4, 6)
    torch.testing.assert_close(A(x), torch.stack([F(x), Id(x)], 0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_stack_puts_the_new_axis_where_it_was_asked(pair, axis):
    F, Id = pair
    x = _rand(*SHAPE)
    A = linop.stack([F, Id], axis=axis)
    torch.testing.assert_close(A(x), torch.stack([F(x), Id(x)], axis), rtol=1e-4, atol=1e-4)


def test_hstack_splits_the_input_and_adds_what_comes_back(pair):
    """[A B] @ [x; y] = A x + B y, built out of the adjoint of a concatenate."""
    F, Id = pair
    xy = _rand(8, 6)
    A = linop.hstack([F, Id])
    assert A.ishape == (8, 6) and A.oshape == SHAPE
    torch.testing.assert_close(A(xy), F(xy[:4]) + Id(xy[4:]), rtol=1e-4, atol=1e-4)


def test_block_diag_gives_each_its_own_part(pair):
    F, Id = pair
    xy = _rand(8, 6)
    A = linop.block_diag([F, Id])
    assert A.ishape == (8, 6) and A.oshape == (8, 6)
    torch.testing.assert_close(A(xy), torch.cat([F(xy[:4]), Id(xy[4:])], 0), rtol=1e-4, atol=1e-4)


def test_block_is_rows_of_hstacks(pair):
    F, Id = pair
    xy = _rand(8, 6)
    A = linop.block([[F, Id], [Id, F]])
    want = torch.cat([F(xy[:4]) + Id(xy[4:]), Id(xy[:4]) + F(xy[4:])], 0)
    torch.testing.assert_close(A(xy), want, rtol=1e-4, atol=1e-4)


def test_one_operator_is_returned_as_it_is(pair):
    F, _ = pair
    assert linop.concatenate([F]) is F


# --- indexing ----------------------------------------------------------------


def test_indexing_restricts_the_output(pair):
    F, _ = pair
    x = _rand(*SHAPE)
    for key in (0, -1, (slice(None), slice(1, 4)), (Ellipsis, 2), (slice(1, 3), slice(0, 2))):
        A = F[key]
        torch.testing.assert_close(A(x), F(x)[key], rtol=1e-4, atol=1e-4)


def test_an_integer_index_drops_the_axis_as_it_does_for_a_tensor(pair):
    F, _ = pair
    assert F[0].oshape == (6,)
    assert F[:, 0].oshape == (4,)


def test_indexing_every_axis_leaves_one_of_size_one(pair):
    """BART has no rank-zero operator, so a scalar comes back as a one."""
    F, _ = pair
    x = _rand(*SHAPE)
    A = F[1, 2]
    assert A.oshape == (1,)
    torch.testing.assert_close(A(x).reshape(()), F(x)[1, 2], rtol=1e-4, atol=1e-4)


def test_the_adjoint_of_a_slice_puts_the_block_back(pair):
    F, _ = pair
    A = F[:, 1:4]
    y = _rand(4, 3)
    back = A.H(y)
    assert back.shape == SHAPE
    torch.testing.assert_close(A(back), A(A.H(y)), rtol=1e-4, atol=1e-4)


def test_a_step_is_refused_rather_than_emulated(pair):
    F, _ = pair
    with pytest.raises(IndexError, match="step other than one"):
        _ = F[::2]


def test_indexing_says_what_it_does_not_take(pair):
    F, _ = pair
    with pytest.raises(IndexError, match="out of range"):
        _ = F[9]
    with pytest.raises(IndexError, match="indices for an operator"):
        _ = F[0, 0, 0]
    with pytest.raises(IndexError, match="only one"):
        _ = F[..., ...]
    with pytest.raises(IndexError, match="integers, slices"):
        _ = F[[0, 1]]


# --- what they have to hold --------------------------------------------------


def _combined():
    F, Id = linop.FFT(SHAPE, axes=-1), linop.Identity(SHAPE)
    return [
        linop.concatenate([F, Id]),
        linop.concatenate([F, Id, F], axis=1),
        linop.stack([F, Id]),
        linop.stack([F, Id], axis=1),
        linop.hstack([F, Id]),
        linop.block_diag([F, Id]),
        linop.block_diag([F, Id, F]),
        linop.block([[F, Id], [Id, F]]),
        F[0],
        F[:, 1:4],
    ]


@pytest.mark.parametrize("A", _combined(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_ones_adjoint_is_its_adjoint(A):
    assert _adjointness(A) < 1e-4


@pytest.mark.parametrize("A", _combined(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_stays_one_bart_operator(A):
    """A stack is one operator a solver drives, not a list walked per iteration."""
    assert A._native
    assert hasattr(A._bart(), "_h")


@pytest.mark.parametrize("A", _combined(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_carries_on_into_the_algebra(A):
    x = _rand(*A.ishape)
    torch.testing.assert_close((2.0 * A.H @ A)(x), 2.0 * A.adjoint(A(x)), rtol=1e-4, atol=1e-4)


def test_a_stack_is_solved_by_barts_conjugate_gradients(pair):
    """Which is the reason for it being one operator rather than a list."""
    from bartorch.optim import CG

    F, Id = pair
    A = linop.concatenate([F, Id])
    x = _rand(*SHAPE)
    got = CG(0.0, maxiter=200, tol=1e-10)(A(x), A, None)
    torch.testing.assert_close(got, x, rtol=1e-2, atol=1e-2)


# --- what they refuse --------------------------------------------------------


def test_concatenate_needs_one_domain(pair):
    F, _ = pair
    with pytest.raises(ValueError, match="one domain for all of them"):
        linop.concatenate([F, linop.Identity((2, 6))])


def test_concatenate_needs_codomains_that_agree_off_the_axis(pair):
    F, _ = pair
    other = linop.Reshape((6, 4), SHAPE) @ F
    with pytest.raises(ValueError, match="must agree off axis"):
        linop.concatenate([F, other])


def test_stack_needs_the_same_shapes_throughout(pair):
    F, _ = pair
    with pytest.raises(ValueError, match="same shapes throughout"):
        linop.stack([F, linop.Reshape((6, 4), SHAPE) @ F])


def test_they_need_at_least_one_operator():
    for fn in (linop.concatenate, linop.stack, linop.hstack, linop.block_diag):
        with pytest.raises(ValueError, match="at least one operator"):
            fn([])
    with pytest.raises(ValueError, match="at least one operator"):
        linop.block([])


def test_they_take_operators_and_not_tensors(pair):
    F, _ = pair
    with pytest.raises(TypeError, match="takes linear operators"):
        linop.concatenate([F, torch.zeros(4, 6)])


def test_block_needs_square_rows(pair):
    F, Id = pair
    with pytest.raises(ValueError, match="same number of operators"):
        linop.block([[F, Id], [Id]])
