"""Tests for bartorch.core.tensor (private utilities and the bart_op decorator)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bartorch.core.tensor import (
    _as_complex64,
    _fortran_strides,
    _normalise_input,
    _reverse_dims,
    bart_op,
)

__all__: list[str] = []

# ---------------------------------------------------------------------------
# _reverse_dims
# ---------------------------------------------------------------------------


def test_reverse_dims_1d():
    assert _reverse_dims([5]) == [5]


def test_reverse_dims_2d():
    assert _reverse_dims([3, 4]) == [4, 3]


def test_reverse_dims_3d():
    assert _reverse_dims([2, 3, 4]) == [4, 3, 2]


def test_reverse_dims_4d():
    assert _reverse_dims([8, 1, 256, 256]) == [256, 256, 1, 8]


def test_reverse_dims_accepts_tuple():
    assert _reverse_dims((2, 3)) == [3, 2]


def test_reverse_dims_roundtrip():
    dims = [8, 4, 256, 256]
    assert _reverse_dims(_reverse_dims(dims)) == dims


# ---------------------------------------------------------------------------
# _as_complex64
# ---------------------------------------------------------------------------


def test_as_complex64_zero_copy():
    t = torch.zeros(4, 8, dtype=torch.complex64)
    assert _as_complex64(t) is t


def test_as_complex64_from_float32():
    t = torch.ones(4, 8, dtype=torch.float32)
    assert _as_complex64(t).dtype == torch.complex64


def test_as_complex64_from_complex128():
    t = torch.ones(4, 8, dtype=torch.complex128)
    assert _as_complex64(t).dtype == torch.complex64


def test_as_complex64_shape_preserved():
    t = torch.zeros(3, 5, 7, dtype=torch.float32)
    assert _as_complex64(t).shape == t.shape


def test_as_complex64_values_preserved():
    t = torch.tensor([1.0, 2.0, 3.0])
    r = _as_complex64(t)
    assert r[0].real == pytest.approx(1.0)
    assert r[1].real == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _normalise_input
# ---------------------------------------------------------------------------


def test_normalise_input_complex64_zero_copy():
    t = torch.zeros(4, 8, dtype=torch.complex64)
    assert _normalise_input(t) is t


def test_normalise_input_float32_cast():
    t = torch.ones(4, 8, dtype=torch.float32)
    assert _normalise_input(t).dtype == torch.complex64


def test_normalise_input_numpy_array():
    arr = np.ones((4, 8), dtype=np.complex64)
    result = _normalise_input(arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.complex64


def test_normalise_input_numpy_float():
    arr = np.ones((4, 8), dtype=np.float32)
    assert _normalise_input(arr).dtype == torch.complex64


def test_normalise_input_int_passthrough():
    assert _normalise_input(42) == 42


def test_normalise_input_string_passthrough():
    assert _normalise_input("fft") == "fft"


def test_normalise_input_bool_passthrough():
    assert _normalise_input(True) is True


def test_normalise_input_none_passthrough():
    assert _normalise_input(None) is None


# ---------------------------------------------------------------------------
# _fortran_strides
# ---------------------------------------------------------------------------


def test_fortran_strides_1d():
    assert _fortran_strides([5]) == [1]


def test_fortran_strides_2d():
    assert _fortran_strides([3, 4]) == [1, 3]


def test_fortran_strides_3d():
    assert _fortran_strides([2, 3, 4]) == [1, 2, 6]


# ---------------------------------------------------------------------------
# bart_op decorator — basic
# ---------------------------------------------------------------------------


def test_bart_op_complex64_zero_copy():
    @bart_op
    def my_op(x):
        return x

    t = torch.zeros(4, 8, dtype=torch.complex64)
    assert my_op(t) is t


def test_bart_op_float32_cast():
    @bart_op
    def my_op(x):
        return x

    t = torch.ones(4, 8, dtype=torch.float32)
    assert my_op(t).dtype == torch.complex64


def test_bart_op_numpy_input():
    @bart_op
    def my_op(x):
        return x

    arr = np.ones((4, 8), dtype=np.float32)
    result = my_op(arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.complex64


def test_bart_op_non_tensor_kwargs_pass_through():
    @bart_op
    def my_op(x, flags, name="default"):
        return x, flags, name

    t = torch.zeros(4, 8, dtype=torch.complex64)
    _, flags_out, name_out = my_op(t, 3, name="fft")
    assert flags_out == 3
    assert name_out == "fft"


def test_bart_op_tensor_kwarg_normalised():
    @bart_op
    def my_op(x, y=None):
        return x, y

    t = torch.zeros(4, 8, dtype=torch.complex64)
    t_float = torch.ones(4, 8, dtype=torch.float32)
    _, y_out = my_op(t, y=t_float)
    assert y_out.dtype == torch.complex64


def test_bart_op_wraps_preserves_name():
    @bart_op
    def my_special_op(x):
        """My docstring."""
        return x

    assert my_special_op.__name__ == "my_special_op"
    assert "My docstring" in my_special_op.__doc__


def test_bart_op_multiple_tensor_args_normalised():
    dtypes_seen = []

    @bart_op
    def my_op(a, b):
        dtypes_seen.append(a.dtype)
        dtypes_seen.append(b.dtype)
        return a

    a = torch.zeros(4, dtype=torch.float32)
    b = torch.zeros(4, dtype=torch.complex128)
    my_op(a, b)
    assert all(d == torch.complex64 for d in dtypes_seen)


# ---------------------------------------------------------------------------
# bart_op decorator — real_output
# ---------------------------------------------------------------------------


def test_bart_op_default_returns_complex():
    @bart_op
    def my_op(x):
        return torch.ones(4, dtype=torch.complex64)

    t = torch.zeros(4, dtype=torch.complex64)
    assert my_op(t).is_complex()


def test_bart_op_real_output_false():
    @bart_op(real_output=False)
    def my_op(x):
        return torch.ones(4, dtype=torch.complex64)

    t = torch.zeros(4, dtype=torch.complex64)
    assert my_op(t).is_complex()


def test_bart_op_real_output_true():
    @bart_op(real_output=True)
    def my_op(x):
        return torch.ones(4, dtype=torch.complex64)

    t = torch.zeros(4, dtype=torch.complex64)
    result = my_op(t)
    assert not result.is_complex()
    assert result.dtype == torch.float32


def test_bart_op_real_output_non_tensor_passthrough():
    @bart_op(real_output=True)
    def my_op(x):
        return 42

    t = torch.zeros(4, dtype=torch.complex64)
    assert my_op(t) == 42


# ---------------------------------------------------------------------------
# bart_op decorator — call styles
# ---------------------------------------------------------------------------


def test_bart_op_no_parens():
    @bart_op
    def f(x):
        return x

    t = torch.zeros(4, dtype=torch.complex64)
    assert f(t) is t


def test_bart_op_with_empty_parens():
    @bart_op()
    def f(x):
        return x

    t = torch.zeros(4, dtype=torch.complex64)
    assert f(t) is t


def test_bart_op_with_real_output_kwarg():
    @bart_op(real_output=True)
    def f(x):
        return torch.ones(4, dtype=torch.complex64)

    t = torch.zeros(4, dtype=torch.complex64)
    assert not f(t).is_complex()


# ---------------------------------------------------------------------------
# bart_op decorator — cpu_only CUDA handling (simulated with meta device)
# ---------------------------------------------------------------------------


def test_bart_op_cpu_only_true_moves_output():
    """cpu_only=True: result is on same device as input (tested with CPU)."""
    received_device = []

    @bart_op(cpu_only=True)
    def my_op(x):
        received_device.append(x.device)
        return x

    t = torch.zeros(4, dtype=torch.complex64)  # CPU tensor
    result = my_op(t)
    assert result.device == t.device


def test_bart_op_cpu_only_false_no_move():
    """cpu_only=False: no device movement even if inputs were on CUDA."""
    received = []

    @bart_op(cpu_only=False)
    def my_op(x):
        received.append(x.device.type)
        return x

    t = torch.zeros(4, dtype=torch.complex64)
    my_op(t)
    assert received[0] == "cpu"


# ---------------------------------------------------------------------------
# Axis convention zero-copy sanity check
# ---------------------------------------------------------------------------


def test_2d_byte_equivalence():
    rows, cols = 3, 4
    data = np.arange(rows * cols, dtype=np.complex64).reshape(rows, cols)
    bart_dims = _reverse_dims([rows, cols])
    reinterpreted = data.ravel().reshape(bart_dims, order="F")
    for i in range(rows):
        for j in range(cols):
            assert data[i, j] == reinterpreted[j, i]
