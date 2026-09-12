"""MRI encoding operators.

Each is a composition of operators BART already has, so what is checked is
that the composition is the encoding it claims to be -- against the operator
it is built on where there is one, and against the chain written out by hand
where there is not -- and that it stays a single BART operator.
"""

import pytest
import torch

from bartorch import linop
from bartorch.optim import CG

COILS, Y, X = 4, 16, 12


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _adjointness(A, seed=0):
    torch.manual_seed(seed)
    x, y = _rand(*A.ishape), _rand(*A.oshape)
    left = torch.vdot(A(x).flatten(), y.flatten())
    right = torch.vdot(x.flatten(), A.H(y).flatten())
    return (left - right).abs().item() / max(left.abs().item(), 1e-12)


@pytest.fixture
def maps():
    torch.manual_seed(0)
    return _rand(COILS, Y, X)


# --- Cartesian SENSE ---------------------------------------------------------


def test_without_a_pattern_it_is_barts_own_operator(maps):
    """There is nothing to add, so nothing is added."""
    A = linop.CartesianSense(maps, (COILS, Y, X))
    assert isinstance(A, linop.Sense)


def test_a_pattern_keeps_the_samples_that_were_taken(maps):
    """Checked against Sense, which this composes rather than replaces."""
    mask = (torch.rand(1, 1, Y, 1) > 0.5).to(torch.complex64)
    S = linop.Sense(maps, (COILS, Y, X))
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=mask)
    x = _rand(*S.ishape)
    torch.testing.assert_close(A(x), mask * S(x), rtol=1e-4, atol=1e-4)


def test_it_keeps_the_shapes_sense_uses(maps):
    """So one can be swapped for the other without reshaping anything."""
    mask = (torch.rand(1, 1, Y, 1) > 0.5).to(torch.complex64)
    S = linop.Sense(maps, (COILS, Y, X))
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=mask)
    assert (A.ishape, A.oshape) == (S.ishape, S.oshape)


def test_a_trajectory_is_refused_here(maps):
    with pytest.raises(ValueError, match="use Sense for that"):
        linop.CartesianSense(maps, (COILS, Y, X), traj=_rand(3, 8, 16))


def test_a_fully_sampled_encoding_inverts_back_to_the_image(maps):
    mask = torch.ones(1, 1, Y, 1, dtype=torch.complex64)
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=mask)
    x = _rand(*A.ishape)
    got = CG(0.0, maxiter=60, tol=1e-8)(A(x), A, None)
    torch.testing.assert_close(got, x, rtol=1e-2, atol=1e-2)


# --- wave --------------------------------------------------------------------

Z, SX, WX = 1, 6, 12
SHAPE = (COILS, Z, 5, SX)


@pytest.fixture
def wave_parts():
    torch.manual_seed(0)
    return (
        _rand(COILS, Z, 5, SX),
        _rand(1, Z, 5, WX),
        (torch.rand(1, Z, 5, WX) > 0.4).to(torch.complex64),
    )


def _by_hand(maps, psf, mask, image, centred):
    """wave.c's chain, written out: E, R, Fx, W, Fyz, M."""
    out = maps * image
    width = out.shape[-1]
    start = abs(WX // 2 - width // 2)
    grown = torch.zeros(*out.shape[:-1], WX, dtype=out.dtype)
    grown[..., start : start + width] = out
    fft = (
        (
            lambda t, d: torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(t, dim=d), dim=d, norm="ortho"), dim=d
            )
        )
        if centred
        else (lambda t, d: torch.fft.fftn(t, dim=d))
    )
    out = fft(grown, (-1,))
    out = psf * out
    out = fft(out, (-3, -2))
    return mask * out


@pytest.mark.parametrize("centred", [False, True])
def test_wave_is_the_chain_bart_builds(wave_parts, centred):
    maps, psf, mask = wave_parts
    A = linop.Wave(maps, psf, SHAPE, readout=WX, pattern=mask, centred=centred)
    image = _rand(*A.ishape)
    torch.testing.assert_close(
        A(image), _by_hand(maps, psf, mask, image, centred), rtol=1e-4, atol=1e-4
    )


def test_wave_oversamples_the_readout_and_leaves_the_rest(wave_parts):
    maps, psf, mask = wave_parts
    A = linop.Wave(maps, psf, SHAPE, readout=WX, pattern=mask)
    assert A.ishape == (Z, 5, SX)
    assert A.oshape == (COILS, Z, 5, WX)


def test_wave_without_a_pattern_is_the_encoding_without_one(wave_parts):
    maps, psf, _ = wave_parts
    A = linop.Wave(maps, psf, SHAPE, readout=WX)
    assert A.oshape == (COILS, Z, 5, WX)
    image = _rand(*A.ishape)
    ones = torch.ones(1, Z, 5, WX, dtype=torch.complex64)
    torch.testing.assert_close(
        A(image), _by_hand(maps, psf, ones, image, False), rtol=1e-4, atol=1e-4
    )


def test_a_readout_shorter_than_the_image_is_refused(wave_parts):
    maps, psf, mask = wave_parts
    with pytest.raises(ValueError, match="shorter than the image"):
        linop.Wave(maps, psf, SHAPE, readout=SX - 1, pattern=mask)


def test_the_wrong_number_of_sensitivities_is_refused(wave_parts):
    _, psf, mask = wave_parts
    with pytest.raises(ValueError, match="sensitivities for"):
        linop.Wave(_rand(2, Z, 5, SX), psf, SHAPE, readout=WX, pattern=mask)


# --- what both have to hold --------------------------------------------------


def _encodings():
    torch.manual_seed(0)
    maps2d = _rand(COILS, Y, X)
    mask2d = (torch.rand(1, 1, Y, 1) > 0.5).to(torch.complex64)
    maps3d = _rand(COILS, Z, 5, SX)
    psf = _rand(1, Z, 5, WX)
    mask = (torch.rand(1, Z, 5, WX) > 0.4).to(torch.complex64)
    return [
        linop.CartesianSense(maps2d, (COILS, Y, X), pattern=mask2d),
        linop.Wave(maps3d, psf, SHAPE, readout=WX, pattern=mask),
        linop.Wave(maps3d, psf, SHAPE, readout=WX, centred=True),
    ]


@pytest.mark.parametrize("A", _encodings(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_ones_adjoint_is_its_adjoint(A):
    assert _adjointness(A) < 1e-4


@pytest.mark.parametrize("A", _encodings(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_is_one_bart_operator(A):
    """Six operators at construction, one in the solver's loop."""
    assert A._native
    assert hasattr(A._bart(), "_h")
    assert A.gram()._native


@pytest.mark.parametrize("A", _encodings(), ids=lambda A: f"{A.ishape}->{A.oshape}")
def test_each_carries_on_into_the_algebra(A):
    x = _rand(*A.ishape)
    torch.testing.assert_close((2.0 * A.H @ A)(x), 2.0 * A.adjoint(A(x)), rtol=1e-4, atol=1e-4)
