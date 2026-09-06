"""
tests/test_cuda.py — CUDA tensor zero-copy tests for bartorch.

All tests are skipped automatically when no CUDA device is available.
To run locally:
    pytest tests/test_cuda.py -v

To run on a GPU CI runner:
    pytest tests/test_cuda.py -v --tb=short
"""

import pytest
import torch

# Skip entire module when CUDA is unavailable.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA device available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bart_available() -> bool:
    """Return True if the bartorch C++ extension compiled and loads correctly."""
    try:
        import bartorch.tools as bt

        bt.phantom(x=8)
        return True
    except Exception:
        return False


bart_available = pytest.mark.skipif(
    not _bart_available(),
    reason="bartorch C++ extension not available",
)

DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Basic CUDA tensor round-trip via BART FFT
# ---------------------------------------------------------------------------


@bart_available
def test_fft_cuda_output_on_device():
    """Output of bt.fft on a CUDA input must be a CUDA tensor."""
    import bartorch.tools as bt

    x = torch.ones(1, 16, 16, dtype=torch.complex64, device=DEVICE)
    y = bt.fft(x, axes=(-1, -2))

    assert y.is_cuda, "bt.fft on CUDA input must return CUDA tensor"
    assert y.device.index == x.device.index
    assert y.shape == x.shape
    assert y.dtype == torch.complex64


@bart_available
def test_fft_cuda_matches_cpu():
    """bt.fft on CUDA tensor must produce the same result as on CPU."""
    import bartorch.tools as bt

    torch.manual_seed(42)
    x_cpu = torch.randn(1, 16, 16, dtype=torch.complex64)
    x_gpu = x_cpu.to(DEVICE)

    y_cpu = bt.fft(x_cpu, axes=(-1, -2))
    y_gpu = bt.fft(x_gpu, axes=(-1, -2))

    # Results must agree to within float32 rounding.
    assert torch.allclose(y_gpu.cpu(), y_cpu, atol=1e-4, rtol=1e-4), (
        f"CUDA/CPU FFT mismatch: max_diff={(y_gpu.cpu() - y_cpu).abs().max().item():.6f}"
    )


@bart_available
def test_ifft_cuda_roundtrip():
    """IFFT(FFT(x)) ≈ x for CUDA tensor."""
    import bartorch.tools as bt

    torch.manual_seed(7)
    x = torch.randn(1, 8, 8, dtype=torch.complex64, device=DEVICE)

    y = bt.ifft(bt.fft(x, axes=(-1, -2)), axes=(-1, -2))

    assert y.is_cuda
    assert torch.allclose(y, x, atol=1e-4, rtol=1e-4), (
        f"CUDA FFT round-trip error: {(y - x).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Mixed device: CPU input must produce CPU output (no silent device change)
# ---------------------------------------------------------------------------


@bart_available
def test_fft_cpu_input_stays_cpu():
    """bt.fft on a CPU tensor must return a CPU tensor, not CUDA."""
    import bartorch.tools as bt

    x = torch.ones(1, 8, 8, dtype=torch.complex64)  # CPU
    y = bt.fft(x, axes=(-1, -2))

    assert not y.is_cuda, "bt.fft on CPU input must stay on CPU"


# ---------------------------------------------------------------------------
# Zero-copy: input tensor data pointer must not be modified by BART
# ---------------------------------------------------------------------------


@bart_available
def test_cuda_input_not_corrupted():
    """BART ops must not modify the original CUDA input tensor in-place."""
    import bartorch.tools as bt

    torch.manual_seed(99)
    x = torch.randn(1, 8, 8, dtype=torch.complex64, device=DEVICE)
    x_orig = x.clone()

    _y = bt.fft(x, axes=(-1, -2))

    assert torch.equal(x, x_orig), "BART modified CUDA input tensor in-place"


# ---------------------------------------------------------------------------
# Phantom on CUDA: output produced by a no-input CUDA-mode call
# (phantom has no tensor inputs, but the output should still be on CPU
# since there are no CUDA inputs to trigger GPU dispatch)
# ---------------------------------------------------------------------------


@bart_available
def test_phantom_cpu_output():
    """bt.phantom() has no CUDA inputs → output is always on CPU."""
    import bartorch.tools as bt

    ph = bt.phantom(x=16)
    assert not ph.is_cuda, "phantom with no CUDA inputs should return CPU tensor"
