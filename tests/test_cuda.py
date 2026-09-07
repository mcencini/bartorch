"""The CUDA surface.

Most of this runs without a device: the API has to report honestly that there
is none and refuse device tensors with a message that says what to do.  The
tests that need a card skip when there is none.
"""

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch.ops import LinearOperator

requires_cuda = pytest.mark.skipif(
    not bartorch.cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)


def test_the_cuda_surface_agrees_with_how_the_library_was_built():
    assert bartorch.cuda.built() == ("cuda=ON" in bartorch.build_info())


def test_device_count_is_zero_when_the_library_has_no_cuda():
    if not bartorch.cuda.built():
        assert bartorch.cuda.device_count() == 0
        assert not bartorch.cuda.available()


def test_free_memory_reports_minus_one_without_a_device():
    if not bartorch.cuda.available():
        assert bartorch.cuda.free_memory() == -1


def test_a_device_tensor_is_refused_with_a_message_naming_the_remedy():
    if bartorch.cuda.available() or not torch.cuda.is_available():
        pytest.skip("a device is usable here, so the refusal does not apply")
    x = torch.randn(4, 8, dtype=torch.complex64, device="cuda")
    with pytest.raises(ValueError, match="cpu"):
        bt.fft(x, axes=-1)


def test_stream_count_is_rejected_outside_barts_range():
    if not bartorch.cuda.built():
        pytest.skip("no CUDA in this build")
    with pytest.raises(ValueError):
        bartorch.cuda.set_streams(0)
    with pytest.raises(ValueError):
        bartorch.cuda.set_streams(99)


@requires_cuda
def test_a_tool_on_a_device_tensor_returns_a_device_tensor():
    x = torch.randn(4, 32, dtype=torch.complex64, device="cuda")
    y = bt.fft(x, axes=-1)
    assert y.device.type == "cuda"
    assert y.device.index == x.device.index


@requires_cuda
def test_a_tool_gives_the_same_answer_on_the_device_as_on_the_host():
    x = torch.randn(4, 32, dtype=torch.complex64)
    host = bt.fft(x, axes=-1)
    device = bt.fft(x.cuda(), axes=-1)
    torch.testing.assert_close(device.cpu(), host, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_an_operator_applies_on_the_device():
    n = 32
    x = torch.randn(1, n, n, dtype=torch.complex64, device="cuda")
    F = LinearOperator.fft((1, n, n), axes=(-1, -2))
    y = F(x)
    assert y.device.type == "cuda"
    torch.testing.assert_close(y.cpu(), F(x.cpu()), rtol=1e-4, atol=1e-4)


@requires_cuda
def test_more_than_one_stream_can_be_asked_for():
    bartorch.cuda.set_streams(2)
    try:
        x = torch.randn(4, 32, dtype=torch.complex64, device="cuda")
        torch.testing.assert_close(
            bt.fft(x, axes=-1).cpu(), bt.fft(x.cpu(), axes=-1), rtol=1e-4, atol=1e-4
        )
    finally:
        bartorch.cuda.set_streams(1)


@requires_cuda
def test_work_is_ordered_against_torchs_stream_without_synchronising():
    # Queue torch work, then BART work, then read back: the result must see
    # the torch work, which it only does if the streams were ordered.
    n = 64
    x = torch.ones(1, n, n, dtype=torch.complex64, device="cuda")
    for _ in range(50):
        x = x * 1.01
    y = bt.fft(x, axes=(-1, -2))
    torch.testing.assert_close(y.cpu(), bt.fft(x.cpu(), axes=(-1, -2)), rtol=1e-3, atol=1e-3)
