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


@requires_cuda
def test_every_tool_kept_on_the_card_answers_there_and_agrees_with_the_host():
    """The list is a claim about each tool's own code, so it is run, not read.

    BART's ``md_`` operations take the host path unless every argument is on a
    device, and take it silently, so a tool that mixes a host temporary with
    the memory it was handed reads device memory from the host.  Whether one
    does is not something to infer from the source; adding a name to
    ``_ON_DEVICE`` without it passing here is how a segmentation fault gets in.
    """
    from bartorch.core.graph import _ON_DEVICE

    # Enough spokes that `pics` is not solving an ill-posed problem: at a
    # thousandth, conjugate gradients over a heavily undersampled radial set
    # amplify the difference between two libraries into tens of percent, which
    # says nothing about whether the tool ran on the card.
    n, coils, spokes = 32, 2, 64
    torch.manual_seed(0)
    img = bt.phantom([n, n], ncoils=coils)
    ksp_cart = bt.fft(img, axes=(-2, -1))
    traj = bt.traj(x=n, y=spokes, r=True)
    ksp_rad = bt.nufft(traj, img)
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    # The two sides are FINUFFT and cuFINUFFT, each promised the tolerance
    # the plans were made with, so what they can differ from each other by is
    # about that.  A solve compounds it and is held looser still.
    cases = {
        "fft": (lambda d: bt.fft(ksp_cart.to(d), axes=(-2, -1)), 1e-5),
        "ifft": (lambda d: bt.ifft(ksp_cart.to(d), axes=(-2, -1)), 1e-5),
        "rss": (lambda d: bt.rss(img.to(d), axes=0), 1e-5),
        "nufft": (lambda d: bt.nufft(traj.to(d), img.to(d)), 5 * bartorch.finufft.tolerance()),
        "pics": (lambda d: bt.pics(ksp_rad.to(d), maps.to(d), t=traj.to(d)), 1e-1),
        "estdims": (lambda d: bt.estdims(traj.to(d)), 0.0),
    }
    assert set(cases) == set(_ON_DEVICE), "every tool kept on the card needs a case here"

    for name, (run, tol) in cases.items():
        on_card, on_host = run("cuda"), run("cpu")
        if not isinstance(on_card, torch.Tensor):
            assert on_card == on_host, name
            continue
        assert on_card.device.type == "cuda", name
        scale = max(float(on_host.abs().max()), 1e-30)
        assert float((on_card.cpu() - on_host).abs().max()) / scale < tol, name


@requires_cuda
def test_a_tool_that_is_not_kept_on_the_card_still_answers_on_it():
    """Crossing to the host is where a tool runs, not what the caller sees."""
    from bartorch.core.graph import _ON_DEVICE

    assert "pocsense" not in _ON_DEVICE
    n, coils = 32, 2
    img = bt.phantom([n, n], ncoils=coils)
    ksp = bt.fft(img, axes=(-2, -1)).cuda()
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()

    out = bt.pocsense(ksp, maps)
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), bt.pocsense(ksp.cpu(), maps.cpu()), rtol=1e-4, atol=1e-5)
