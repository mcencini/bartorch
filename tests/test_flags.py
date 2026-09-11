"""C-order axes to BART bitmasks (``bartorch._flags``)."""

from __future__ import annotations

import pytest

from bartorch._flags import _axes_to_flags as axes_to_flags
from bartorch._flags import _indices_to_flags as indices_to_flags

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_single_last_axis_ndim2():
    assert axes_to_flags(1, ndim=2) == 1


def test_single_first_axis_ndim2():
    assert axes_to_flags(0, ndim=2) == 2


def test_both_axes_ndim2():
    assert axes_to_flags((0, 1), ndim=2) == 3


def test_last_two_axes_ndim3():
    assert axes_to_flags((1, 2), ndim=3) == 3


def test_first_axis_ndim3():
    assert axes_to_flags(0, ndim=3) == 4


def test_all_axes_ndim3():
    assert axes_to_flags((0, 1, 2), ndim=3) == 7


# ---------------------------------------------------------------------------
# Negative indices
# ---------------------------------------------------------------------------


def test_minus1_equiv_last():
    for ndim in range(1, 6):
        assert axes_to_flags(-1, ndim=ndim) == axes_to_flags(ndim - 1, ndim=ndim)


def test_minus2_equiv_second_to_last():
    for ndim in range(2, 6):
        assert axes_to_flags(-2, ndim=ndim) == axes_to_flags(ndim - 2, ndim=ndim)


def test_negative_tuple_equiv_positive():
    assert axes_to_flags((-1, -2), ndim=3) == axes_to_flags((1, 2), ndim=3)


def test_all_negative():
    assert axes_to_flags((-1, -2, -3), ndim=3) == 7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_ndim1_single_axis():
    assert axes_to_flags(0, ndim=1) == 1
    assert axes_to_flags(-1, ndim=1) == 1


def test_list_input():
    assert axes_to_flags([0, 1], ndim=2) == 3


def test_single_int_not_tuple():
    assert axes_to_flags(0, ndim=3) == 4


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_ndim_zero_raises():
    with pytest.raises(ValueError, match="ndim"):
        axes_to_flags(0, ndim=0)


def test_axis_too_large_raises():
    with pytest.raises(ValueError):
        axes_to_flags(3, ndim=3)


def test_negative_axis_too_small_raises():
    with pytest.raises(ValueError):
        axes_to_flags(-4, ndim=3)


def test_duplicate_axes_raises():
    with pytest.raises(ValueError, match="duplicate"):
        axes_to_flags((0, 0), ndim=2)


def test_duplicate_mixed_sign_raises():
    with pytest.raises(ValueError, match="duplicate"):
        axes_to_flags((1, -1), ndim=2)


# ---------------------------------------------------------------------------
# Consistency / roundtrip
# ---------------------------------------------------------------------------


def test_typical_2d_fft():
    assert axes_to_flags((0, 1), ndim=2) == 3


def test_typical_3d_fft_last_two():
    assert axes_to_flags((-1, -2), ndim=3) == 3


def test_roundtrip_all_axes():
    for ndim in range(1, 5):
        expected = (1 << ndim) - 1
        assert axes_to_flags(list(range(ndim)), ndim=ndim) == expected
        assert axes_to_flags(list(range(-ndim, 0)), ndim=ndim) == expected


# ---------------------------------------------------------------------------
# No array to count from, and index sets
# ---------------------------------------------------------------------------


def test_a_negative_axis_needs_no_array():
    """Axis -1 is BART's dimension 0 whatever the number of axes."""
    assert axes_to_flags((-1, -3), ndim=None) == axes_to_flags((-1, -3), ndim=5) == 5


def test_a_non_negative_axis_needs_an_array():
    with pytest.raises(ValueError, match="negative"):
        axes_to_flags(0, ndim=None)


def test_an_index_set_is_not_reversed():
    assert indices_to_flags((0, 2)) == 5
    assert indices_to_flags(3) == 8


def test_an_index_set_refuses_what_is_not_an_index():
    with pytest.raises(ValueError, match="repeated"):
        indices_to_flags((1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        indices_to_flags(-1)
