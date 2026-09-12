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
    assert isinstance(A, linop.NoncartesianSense)


def test_a_pattern_keeps_the_samples_that_were_taken(maps):
    """Checked against the encoding alone, which this composes rather than replaces."""
    mask = (torch.rand(1, 1, Y, 1) > 0.5).to(torch.complex64)
    S = linop.CartesianSense(maps, (COILS, Y, X))
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=mask)
    x = _rand(*S.ishape)
    torch.testing.assert_close(A(x), mask * S(x), rtol=1e-4, atol=1e-4)


def test_it_keeps_the_shapes_sense_uses(maps):
    """So one can be swapped for the other without reshaping anything."""
    mask = (torch.rand(1, 1, Y, 1) > 0.5).to(torch.complex64)
    S = linop.CartesianSense(maps, (COILS, Y, X))
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=mask)
    assert (A.ishape, A.oshape) == (S.ishape, S.oshape)


def test_a_trajectory_is_refused_here(maps):
    with pytest.raises(ValueError, match="use NoncartesianSense for that"):
        linop.CartesianSense(maps, (COILS, Y, X), traj=_rand(3, 8, 16))


def test_the_noncartesian_encoding_wants_a_trajectory(maps):
    """The names say which is which, so neither stands in for the other."""
    with pytest.raises(ValueError, match="CartesianSense is the one on it"):
        linop.NoncartesianSense(maps, (COILS, Y, X))


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
    A = linop.WaveSense(maps, psf, SHAPE, readout=WX, pattern=mask, centred=centred)
    image = _rand(*A.ishape)
    torch.testing.assert_close(
        A(image), _by_hand(maps, psf, mask, image, centred), rtol=1e-4, atol=1e-4
    )


def test_wave_oversamples_the_readout_and_leaves_the_rest(wave_parts):
    maps, psf, mask = wave_parts
    A = linop.WaveSense(maps, psf, SHAPE, readout=WX, pattern=mask)
    assert A.ishape == (Z, 5, SX)
    assert A.oshape == (COILS, Z, 5, WX)


def test_wave_without_a_pattern_is_the_encoding_without_one(wave_parts):
    maps, psf, _ = wave_parts
    A = linop.WaveSense(maps, psf, SHAPE, readout=WX)
    assert A.oshape == (COILS, Z, 5, WX)
    image = _rand(*A.ishape)
    ones = torch.ones(1, Z, 5, WX, dtype=torch.complex64)
    torch.testing.assert_close(
        A(image), _by_hand(maps, psf, ones, image, False), rtol=1e-4, atol=1e-4
    )


def test_a_readout_shorter_than_the_image_is_refused(wave_parts):
    maps, psf, mask = wave_parts
    with pytest.raises(ValueError, match="shorter than the image"):
        linop.WaveSense(maps, psf, SHAPE, readout=SX - 1, pattern=mask)


def test_the_wrong_number_of_sensitivities_is_refused(wave_parts):
    _, psf, mask = wave_parts
    with pytest.raises(ValueError, match="are neither"):
        linop.WaveSense(_rand(2, Z, 5, SX), psf, SHAPE, readout=WX, pattern=mask)


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
        linop.WaveSense(maps3d, psf, SHAPE, readout=WX, pattern=mask),
        linop.WaveSense(maps3d, psf, SHAPE, readout=WX, centred=True),
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


# --- off-resonance -----------------------------------------------------------


def _dense(A, size):
    cols = []
    for i in range(size):
        e = torch.zeros(*A.ishape, dtype=torch.complex64)
        e.reshape(-1)[i] = 1
        cols.append(A(e).reshape(-1).numpy())
    import numpy as np

    return np.stack(cols, axis=1)


def _exact_off_resonance(n, fmap, times):
    """The operator time segmentation approximates: a different phase per sample."""
    import numpy as np
    from mrinufft.extras.field_map import get_complex_fieldmap_rad

    w = np.asarray(get_complex_fieldmap_rad(fmap.squeeze().numpy()))
    dft = np.fft.fft(np.eye(n), axis=0)
    return np.exp(np.outer(times.squeeze().numpy(), w)) * dft


@pytest.fixture
def one_dimensional():
    n = 16
    return (
        n,
        linop.FFT((1, 1, n), axes=-1, centred=False),
        torch.linspace(-120.0, 120.0, n).reshape(1, 1, n),
        torch.linspace(0.0, 4e-3, n).reshape(1, 1, n),
    )


def test_more_segments_is_a_better_approximation(one_dimensional):
    """Which is the whole claim: a short sum standing in for a different transform per sample."""
    import numpy as np

    n, E, fmap, times = one_dimensional
    exact = _exact_off_resonance(n, fmap, times)
    errors = []
    for segments in (1, 2, 4):
        A = linop.FieldCorrected(E, fmap, times, segments=segments)
        errors.append(np.linalg.norm(_dense(A, n) - exact) / np.linalg.norm(exact))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-2, f"four segments left {errors[-1]:.1e}"


def test_almost_no_off_resonance_barely_changes_the_encoding(one_dimensional):
    import numpy as np

    n, E, _, times = one_dimensional
    faint = torch.linspace(-0.05, 0.05, n).reshape(1, 1, n)  # Hz, over 4 ms
    A = linop.FieldCorrected(E, faint, times, segments=2)
    plain, corrected = _dense(E, n), _dense(A, n)
    assert np.linalg.norm(corrected - plain) / np.linalg.norm(plain) < 1e-3


def test_a_field_map_with_one_value_says_what_is_wrong(one_dimensional):
    """mri-nufft's fit bins the map, and one value is one bin; it fails inside
    its own reshape, which is no help to whoever passed it."""
    n, E, _, times = one_dimensional
    with pytest.raises(ValueError, match="nothing to segment"):
        linop.FieldCorrected(E, torch.zeros(1, 1, n), times, segments=1)


def test_it_wraps_any_encoding(maps):
    """Cartesian here; the same wrapper over NoncartesianSense is what mirtorch calls Gmri."""
    E = linop.CartesianSense(
        maps, (COILS, Y, X), pattern=torch.ones(1, 1, Y, 1).to(torch.complex64)
    )
    fmap = torch.linspace(-80.0, 80.0, Y * X).reshape(1, Y, X)
    times = torch.linspace(0.0, 3e-3, Y * X).reshape(1, 1, Y, X).expand(COILS, 1, Y, X)
    A = linop.FieldCorrected(E, fmap, times, segments=3)
    assert A.ishape == E.ishape and A.oshape == E.oshape
    assert A._native, "the sum of chains left BART"
    assert _adjointness(A) < 1e-4


def test_precomputed_coefficients_skip_the_fit(one_dimensional):
    n, E, fmap, times = one_dimensional
    fitted = linop.FieldCorrected(E, fmap, times, segments=3)
    b, c = linop.mri._fit_coefficients(E, fmap, times, None, 3, "svd")
    given = linop.FieldCorrected(E, coefficients=(b, c))
    x = _rand(*E.ishape)
    torch.testing.assert_close(given(x), fitted(x), rtol=1e-4, atol=1e-4)


def test_it_needs_a_field_map_or_coefficients(one_dimensional):
    _, E, _, _ = one_dimensional
    with pytest.raises(ValueError, match="or coefficients"):
        linop.FieldCorrected(E)


def test_mismatched_coefficients_are_refused(one_dimensional):
    _, E, _, _ = one_dimensional
    with pytest.raises(ValueError, match="sample weights against"):
        linop.FieldCorrected(E, coefficients=(_rand(3, 1, 1, 16), _rand(2, 1, 1, 16)))


def test_a_segmented_encoding_is_still_one_bart_operator(one_dimensional):
    """A sum of chains: linop_plus over linop_chain, and nothing in Python."""
    _, E, fmap, times = one_dimensional
    A = linop.FieldCorrected(E, fmap, times, segments=4)
    assert A._native
    assert hasattr(A._bart(), "_h")
    assert A.gram()._native


# --- temporal subspaces on a grid --------------------------------------------
#
# T2 shuffling: the image is a few coefficients, the basis contracts them into
# the frames that were acquired, and the pattern keeps the samples.  The claim
# worth checking is the normal, which collapses the frames into one kernel and
# so never makes them.

FRAMES, COEFFS = 8, 2


@pytest.fixture
def basis():
    torch.manual_seed(1)
    return _rand(COEFFS, FRAMES).reshape(COEFFS, FRAMES, 1, 1, 1, 1, 1)


@pytest.fixture
def frame_pattern():
    torch.manual_seed(2)
    return (torch.rand(1, FRAMES, 1, 1, 1, Y, 1) > 0.4).to(torch.complex64)


def _explicit_subspace(maps, basis, pattern, x):
    """The model written out: coils, transform, basis, pattern."""
    import bartorch

    b = basis.reshape(COEFFS, FRAMES)
    coil_images = maps.reshape(1, COILS, Y, X) * x.reshape(COEFFS, 1, Y, X)
    ksp = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)
    frames = torch.einsum("kt,kcyx->tcyx", b, ksp)
    return frames.reshape(1, FRAMES, 1, COILS, 1, Y, X) * pattern


def test_a_subspace_encoding_is_the_model_written_out(maps, basis, frame_pattern):
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=frame_pattern, basis=basis)

    assert A.ishape == (COEFFS, 1, 1, 1, 1, Y, X)
    assert A.oshape == (1, FRAMES, 1, COILS, 1, Y, X)

    torch.manual_seed(0)
    x = _rand(*A.ishape)
    want = _explicit_subspace(maps, basis, frame_pattern, x)
    torch.testing.assert_close(A(x), want, rtol=1e-4, atol=1e-4)


def test_a_subspace_encoding_has_the_adjoint_it_claims(maps, basis, frame_pattern):
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=frame_pattern, basis=basis)
    assert _adjointness(A) < 1e-5


def test_the_collapsed_normal_is_the_two_applications(maps, basis, frame_pattern):
    """Which is the whole claim: one kernel instead of the frames."""
    fast = linop.CartesianSense(
        maps, (COILS, Y, X), pattern=frame_pattern, basis=basis, toeplitz=True
    )
    slow = linop.CartesianSense(
        maps, (COILS, Y, X), pattern=frame_pattern, basis=basis, toeplitz=False
    )

    torch.manual_seed(0)
    x = _rand(*fast.ishape)

    torch.testing.assert_close(fast(x), slow(x), rtol=0, atol=0)

    want = slow.adjoint(slow(x))
    got = fast.normal(x)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_the_collapsed_normal_does_not_grow_with_the_frames(maps, basis):
    """The kernel is coefficients by coefficients, whatever the echo train.

    Sixty-four frames and two coefficients is the same four numbers per voxel
    as eight frames would be, which is why an iteration costs what it does.
    """
    from bartorch.linop.mri import _subspace_kernel

    for frames in (8, 64, 256):
        torch.manual_seed(3)
        b = _rand(COEFFS, frames)
        pattern = (torch.rand(1, frames, 1, 1, 1, Y, 1) > 0.4).to(torch.complex64)
        kernel = _subspace_kernel(b, pattern, (1, frames, 1, COILS, 1, Y, X))
        assert kernel.shape == (COEFFS, COEFFS, 1, 1, 1, Y, 1)


def test_a_subspace_normal_without_a_pattern_is_the_basis_gram(maps, basis):
    """Every sample taken, so what is left of the sum is the basis alone."""
    A = linop.CartesianSense(maps, (COILS, Y, X), basis=basis, toeplitz=True)
    B = linop.CartesianSense(maps, (COILS, Y, X), basis=basis, toeplitz=False)

    torch.manual_seed(0)
    x = _rand(*A.ishape)
    want = B.adjoint(B(x))
    assert (A.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_subspace_encoding_solves(maps, basis, frame_pattern):
    """And the solver drives the collapsed normal, which is what CG asks for."""
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=frame_pattern, basis=basis)
    torch.manual_seed(0)
    x = _rand(*A.ishape)
    got = CG(maxiter=40)(A(x), A)
    assert (got - x).abs().max() / x.abs().max() < 0.2


def test_a_basis_that_is_not_one_is_refused(maps):
    with pytest.raises(ValueError, match=r"a basis is \(coeffs, frames"):
        linop.CartesianSense(maps, (COILS, Y, X), basis=_rand(COEFFS, FRAMES, 3))


# --- wave, on the coil loop ---------------------------------------------------


def test_wave_takes_the_kernels_nlinv_produces(wave_parts):
    """The same encoding, with the bank inflated a slab at a time."""
    import bartorch

    _, psf, mask = wave_parts
    torch.manual_seed(4)
    kernels = _rand(COILS, Z, 3, 4)
    dense = bartorch.kernels_to_maps(kernels, (Z, 5, SX))

    a = linop.WaveSense(dense, psf, SHAPE, readout=WX, pattern=mask)
    b = linop.WaveSense(kernels, psf, SHAPE, readout=WX, pattern=mask, kernels=True)

    x = _rand(*a.ishape)
    torch.testing.assert_close(b(x), a(x), rtol=1e-4, atol=1e-4)
    y = a(x)
    torch.testing.assert_close(b.adjoint(y), a.adjoint(y), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [0, 1, 2, 4])
def test_the_wave_coil_slab_changes_nothing(wave_parts, batch):
    maps, psf, mask = wave_parts
    A = linop.WaveSense(maps, psf, SHAPE, readout=WX, pattern=mask, coil_batch=batch)
    B = linop.WaveSense(maps, psf, SHAPE, readout=WX, pattern=mask, coil_batch=0)
    x = _rand(*A.ishape)
    torch.testing.assert_close(A(x), B(x), rtol=1e-5, atol=1e-5)


def test_wave_shuffling_is_wave_read_through_a_subspace(wave_parts, basis):
    maps, psf, _ = wave_parts
    A = linop.WaveSense(maps, psf, SHAPE, readout=WX, basis=basis)
    plain = linop.WaveSense(maps, psf, SHAPE, readout=WX)

    assert A.ishape == (COEFFS, 1, 1, 1, *plain.ishape)
    assert A.oshape == (1, FRAMES, 1, *plain.oshape)
    assert _adjointness(A) < 1e-5


def test_the_collapsed_normal_carries_to_wave(wave_parts, basis):
    """The kernel goes where the sampling goes, and it is the same kernel."""
    maps, psf, _ = wave_parts
    torch.manual_seed(5)
    pattern = (torch.rand(1, FRAMES, 1, 1, Z, 5, 1) > 0.4).to(torch.complex64)

    fast = linop.WaveSense(maps, psf, SHAPE, readout=WX, pattern=pattern, basis=basis)
    slow = linop.WaveSense(
        maps, psf, SHAPE, readout=WX, pattern=pattern, basis=basis, toeplitz=False
    )

    x = _rand(*fast.ishape)
    torch.testing.assert_close(fast(x), slow(x), rtol=0, atol=0)

    want = slow.adjoint(slow(x))
    assert (fast.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_collapsed_subspace_encoding_is_still_one_bart_operator(maps, basis, frame_pattern):
    """The normal is composed from operators, so both sides stay BART's."""
    A = linop.CartesianSense(maps, (COILS, Y, X), pattern=frame_pattern, basis=basis)
    assert A._native
    assert hasattr(A._bart(), "_h")
    assert A.gram()._native

    D = linop.Diagonal(_rand(*A.oshape), A.oshape)
    assert (D @ A)._native


# --- several sets of maps -----------------------------------------------------
#
# What ESPIRiT's second map and ENLIVE's relaxed model produce.  The shapes are
# checked in test_sense.py; here it is that each encoding carries them through
# and that a reconstruction over two sets actually runs.

SETS = 2


@pytest.fixture
def bank():
    torch.manual_seed(9)
    return _rand(SETS, COILS, 1, Y, X)


def test_espirit_hands_its_second_map_over_as_it_stands():
    """No reshaping between the calibration and the encoding."""
    import bartorch.tools as bt

    kspace = bt.phantom([Y, Y], coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=SETS)
    assert maps.shape == (SETS, COILS, 1, Y, Y)

    A = linop.CartesianSense(maps, (COILS, Y, Y))
    assert A.ishape == (1, 1, SETS, 1, 1, Y, Y)
    assert A.oshape == (1, 1, 1, COILS, 1, Y, Y)


def test_a_two_map_reconstruction_fits_the_data_it_was_given():
    """Not the image it was given: two sets of ESPIRiT maps do not span an
    arbitrary pair of images, which is the whole point of the second one.  The
    operator has a null space, so what CG converges to is a least-squares
    solution -- and it is the data that has to come back, not the image."""
    import bartorch.tools as bt

    kspace = bt.phantom([Y, Y], coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=SETS)

    A = linop.CartesianSense(maps, (COILS, Y, Y))
    torch.manual_seed(0)
    y = A(_rand(*A.ishape))
    got = CG(maxiter=80)(y, A)
    assert (A(got) - y).abs().max() / y.abs().max() < 1e-2


def test_a_cartesian_encoding_over_sets_is_the_sum_of_the_one_set_ones(bank):
    A = linop.CartesianSense(bank, (COILS, Y, X))
    torch.manual_seed(0)
    x = _rand(*A.ishape)

    want = sum(linop.CartesianSense(bank[m], (COILS, Y, X))(x[0, 0, m, 0]) for m in range(SETS))
    torch.testing.assert_close(A(x).reshape(want.shape), want, rtol=1e-4, atol=1e-5)


def test_a_pattern_and_several_sets_go_together(bank):
    mask = (torch.rand(1, 1, 1, 1, 1, Y, 1) > 0.4).to(torch.complex64)
    A = linop.CartesianSense(bank, (COILS, Y, X), pattern=mask)
    assert _adjointness(A) < 1e-5


def test_a_subspace_and_several_sets_go_together(bank, basis, frame_pattern):
    A = linop.CartesianSense(bank, (COILS, Y, X), pattern=frame_pattern, basis=basis)
    slow = linop.CartesianSense(
        bank, (COILS, Y, X), pattern=frame_pattern, basis=basis, toeplitz=False
    )

    assert A.ishape == (COEFFS, 1, SETS, 1, 1, Y, X)
    assert A.oshape == (1, FRAMES, 1, COILS, 1, Y, X)
    assert _adjointness(A) < 1e-5

    torch.manual_seed(0)
    x = _rand(*A.ishape)
    want = slow.adjoint(slow(x))
    assert (A.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_wave_carries_several_sets_too(wave_parts):
    _, psf, mask = wave_parts
    torch.manual_seed(10)
    sets_bank = _rand(SETS, COILS, Z, 5, SX)

    A = linop.WaveSense(sets_bank, psf, SHAPE, readout=WX, pattern=mask)
    assert A.ishape == (1, 1, SETS, 1, Z, 5, SX)
    assert A.oshape == (1, 1, 1, COILS, Z, 5, WX)
    assert _adjointness(A) < 1e-5
