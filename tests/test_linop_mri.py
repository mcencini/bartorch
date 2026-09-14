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
    A = linop.CartesianSense(maps, (Y, X))
    assert isinstance(A, linop.NoncartesianSense)


def test_a_pattern_keeps_the_samples_that_were_taken(maps):
    """Checked against the encoding alone, which this composes rather than replaces."""
    mask = (torch.rand(Y, 1) > 0.5).to(torch.complex64)
    S = linop.CartesianSense(maps, (Y, X))
    A = linop.CartesianSense(maps, (Y, X), pattern=mask)
    x = _rand(*S.ishape)
    torch.testing.assert_close(A(x), mask * S(x), rtol=1e-4, atol=1e-4)


def test_it_keeps_the_shapes_sense_uses(maps):
    """So one can be swapped for the other without reshaping anything."""
    mask = (torch.rand(Y, 1) > 0.5).to(torch.complex64)
    S = linop.CartesianSense(maps, (Y, X))
    A = linop.CartesianSense(maps, (Y, X), pattern=mask)
    assert (A.ishape, A.oshape) == (S.ishape, S.oshape)


def test_a_trajectory_is_refused_here(maps):
    with pytest.raises(ValueError, match="use NoncartesianSense for that"):
        linop.CartesianSense(maps, (Y, X), traj=_rand(8, 16, 3))


def test_the_noncartesian_encoding_wants_a_trajectory(maps):
    """The names say which is which, so neither stands in for the other."""
    with pytest.raises(ValueError, match="CartesianSense is the one on it"):
        linop.NoncartesianSense(maps, (Y, X))


def test_a_fully_sampled_encoding_inverts_back_to_the_image(maps):
    mask = torch.ones(Y, 1, dtype=torch.complex64)
    A = linop.CartesianSense(maps, (Y, X), pattern=mask)
    x = _rand(*A.ishape)
    got = CG(0.0, maxiter=60, tol=1e-8)(A(x), A, None)
    torch.testing.assert_close(got, x, rtol=1e-2, atol=1e-2)


# --- wave --------------------------------------------------------------------

Z, SX, WX = 1, 6, 12
SHAPE = (Z, 5, SX)


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
    A = linop.WaveSense(maps, SHAPE, psf=psf, readout=WX, pattern=mask, centred=centred)
    image = _rand(*A.ishape)
    torch.testing.assert_close(
        A(image), _by_hand(maps, psf, mask, image, centred), rtol=1e-4, atol=1e-4
    )


def test_wave_oversamples_the_readout_and_leaves_the_rest(wave_parts):
    maps, psf, mask = wave_parts
    A = linop.WaveSense(maps, SHAPE, psf=psf, readout=WX, pattern=mask)
    assert A.ishape == (Z, 5, SX)
    assert A.oshape == (COILS, Z, 5, WX)


def test_wave_without_a_pattern_is_the_encoding_without_one(wave_parts):
    maps, psf, _ = wave_parts
    A = linop.WaveSense(maps, SHAPE, psf=psf, readout=WX)
    assert A.oshape == (COILS, Z, 5, WX)
    image = _rand(*A.ishape)
    ones = torch.ones(1, Z, 5, WX, dtype=torch.complex64)
    torch.testing.assert_close(
        A(image), _by_hand(maps, psf, ones, image, False), rtol=1e-4, atol=1e-4
    )


def test_a_readout_shorter_than_the_image_is_refused(wave_parts):
    maps, psf, mask = wave_parts
    with pytest.raises(ValueError, match="shorter than the image"):
        linop.WaveSense(maps, SHAPE, psf=psf, readout=SX - 1, pattern=mask)


def test_a_bank_that_is_neither_coils_nor_sets_of_them_is_refused(wave_parts):
    """The coils are the bank's to say, so what can be wrong with it is its rank."""
    _, psf, mask = wave_parts
    with pytest.raises(ValueError, match="are neither"):
        linop.WaveSense(_rand(2, 2, COILS, Z, 5, SX), SHAPE, psf=psf, readout=WX, pattern=mask)


# --- what both have to hold --------------------------------------------------


def _encodings():
    torch.manual_seed(0)
    maps2d = _rand(COILS, Y, X)
    mask2d = (torch.rand(Y, 1) > 0.5).to(torch.complex64)
    maps3d = _rand(COILS, Z, 5, SX)
    psf = _rand(1, Z, 5, WX)
    mask = (torch.rand(1, Z, 5, WX) > 0.4).to(torch.complex64)
    return [
        linop.CartesianSense(maps2d, (Y, X), pattern=mask2d),
        linop.WaveSense(maps3d, SHAPE, psf=psf, readout=WX, pattern=mask),
        linop.WaveSense(maps3d, SHAPE, psf=psf, readout=WX, centred=True),
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
    E = linop.CartesianSense(maps, (Y, X), pattern=torch.ones(Y, 1).to(torch.complex64))
    fmap = torch.linspace(-80.0, 80.0, Y * X).reshape(Y, X)
    times = torch.linspace(0.0, 3e-3, Y * X).reshape(1, Y, X).expand(COILS, Y, X)
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
    return _rand(COEFFS, FRAMES)


@pytest.fixture
def frame_pattern():
    torch.manual_seed(2)
    return (torch.rand(FRAMES, Y, 1) > 0.4).to(torch.complex64)


def _explicit_subspace(maps, basis, pattern, x):
    """The model written out: coils, transform, basis, pattern."""
    import bartorch

    coil_images = maps.reshape(1, COILS, Y, X) * x.reshape(COEFFS, 1, Y, X)
    ksp = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)
    frames = torch.einsum("kt,kcyx->ctyx", basis, ksp)
    return frames * pattern


def test_a_subspace_encoding_is_the_model_written_out(maps, basis, frame_pattern):
    A = linop.CartesianSense(maps, (COEFFS, Y, X), pattern=frame_pattern, basis=basis)

    assert A.ishape == (COEFFS, Y, X)
    assert A.oshape == (COILS, FRAMES, Y, X)

    torch.manual_seed(0)
    x = _rand(*A.ishape)
    want = _explicit_subspace(maps, basis, frame_pattern, x)
    torch.testing.assert_close(A(x), want, rtol=1e-4, atol=1e-4)


def test_a_subspace_encoding_has_the_adjoint_it_claims(maps, basis, frame_pattern):
    A = linop.CartesianSense(maps, (COEFFS, Y, X), pattern=frame_pattern, basis=basis)
    assert _adjointness(A) < 1e-5


def test_the_collapsed_normal_is_the_two_applications(maps, basis, frame_pattern):
    """Which is the whole claim: one kernel instead of the frames."""
    fast = linop.CartesianSense(
        maps, (COEFFS, Y, X), pattern=frame_pattern, basis=basis, toeplitz=True
    )
    slow = linop.CartesianSense(
        maps, (COEFFS, Y, X), pattern=frame_pattern, basis=basis, toeplitz=False
    )

    torch.manual_seed(0)
    x = _rand(*fast.ishape)

    torch.testing.assert_close(fast(x), slow(x), rtol=0, atol=0)

    want = slow.adjoint(slow(x))
    got = fast.normal(x)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_the_collapsed_normal_does_not_grow_with_the_frames(maps):
    """The kernel is coefficients by coefficients, whatever the echo train.

    Sixty-four frames and two coefficients is the same four numbers per voxel
    as eight frames would be, which is why an iteration costs what it does.
    The operator's normal is held against that kernel applied between the
    encoding and its adjoint, at echo trains up to thirty-two times longer.
    """
    for frames in (8, 64, 256):
        torch.manual_seed(3)
        b = _rand(COEFFS, frames)
        pattern = (torch.rand(frames, Y, 1) > 0.4).to(torch.complex64)
        A = linop.CartesianSense(maps, (COEFFS, Y, X), pattern=pattern, basis=b)

        # sum_t P[t] conj(B[k', t]) B[k, t], over the axes the pattern varies on.
        kernel = torch.einsum("ty,jt,kt->jky", pattern[..., 0], b.conj(), b)[..., None]
        assert kernel.shape == (COEFFS, COEFFS, Y, 1)

        x = _rand(*A.ishape)
        k = _fftc(maps[None] * x[:, None], (-2, -1))
        back = (kernel[:, :, None] * k[None]).sum(1)
        want = (maps.conj()[None] * _ifftc(back, (-2, -1))).sum(1)
        assert (A.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_subspace_normal_without_a_pattern_is_the_basis_gram(maps, basis):
    """Every sample taken, so what is left of the sum is the basis alone."""
    A = linop.CartesianSense(maps, (COEFFS, Y, X), basis=basis, toeplitz=True)
    B = linop.CartesianSense(maps, (COEFFS, Y, X), basis=basis, toeplitz=False)

    torch.manual_seed(0)
    x = _rand(*A.ishape)
    want = B.adjoint(B(x))
    assert (A.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_subspace_encoding_solves(maps, basis, frame_pattern):
    """And the solver drives the collapsed normal, which is what CG asks for."""
    A = linop.CartesianSense(maps, (COEFFS, Y, X), pattern=frame_pattern, basis=basis)
    torch.manual_seed(0)
    x = _rand(*A.ishape)
    got = CG(maxiter=40)(A(x), A)
    assert (got - x).abs().max() / x.abs().max() < 0.2


def test_a_basis_that_is_not_one_is_refused(maps):
    with pytest.raises(ValueError, match=r"a basis is \(coeffs, frames"):
        linop.CartesianSense(maps, (COEFFS, Y, X), basis=_rand(COEFFS, FRAMES, 3))


# --- wave, on the coil loop ---------------------------------------------------


def test_wave_takes_the_kernels_nlinv_produces(wave_parts):
    """The same encoding, with the bank inflated a slab at a time."""
    import bartorch

    _, psf, mask = wave_parts
    torch.manual_seed(4)
    kernels = _rand(COILS, Z, 3, 4)
    dense = bartorch.kernels_to_maps(kernels, (Z, 5, SX))

    a = linop.WaveSense(dense, SHAPE, psf=psf, readout=WX, pattern=mask)
    b = linop.WaveSense(kernels, SHAPE, psf=psf, readout=WX, pattern=mask, kernels=True, ndim=3)

    x = _rand(*a.ishape)
    torch.testing.assert_close(b(x), a(x), rtol=1e-4, atol=1e-4)
    y = a(x)
    torch.testing.assert_close(b.adjoint(y), a.adjoint(y), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [0, 1, 2, 4])
def test_the_wave_coil_slab_changes_nothing(wave_parts, batch):
    maps, psf, mask = wave_parts
    A = linop.WaveSense(maps, SHAPE, psf=psf, readout=WX, pattern=mask, coil_batch=batch)
    B = linop.WaveSense(maps, SHAPE, psf=psf, readout=WX, pattern=mask, coil_batch=0)
    x = _rand(*A.ishape)
    torch.testing.assert_close(A(x), B(x), rtol=1e-5, atol=1e-5)


def test_wave_shuffling_is_wave_read_through_a_subspace(wave_parts, basis):
    maps, psf, _ = wave_parts
    A = linop.WaveSense(maps, (COEFFS, *SHAPE), psf=psf, readout=WX, basis=basis)
    plain = linop.WaveSense(maps, SHAPE, psf=psf, readout=WX)

    assert A.ishape == (COEFFS, *plain.ishape)
    assert A.oshape == (COILS, FRAMES, *plain.oshape[1:])
    assert _adjointness(A) < 1e-5


def test_the_collapsed_normal_carries_to_wave(wave_parts, basis):
    """The kernel goes where the sampling goes, and it is the same kernel."""
    maps, psf, _ = wave_parts
    torch.manual_seed(5)
    pattern = (torch.rand(FRAMES, Z, 5, 1) > 0.4).to(torch.complex64)

    fast = linop.WaveSense(
        maps, (COEFFS, *SHAPE), psf=psf, readout=WX, pattern=pattern, basis=basis
    )
    slow = linop.WaveSense(
        maps, (COEFFS, *SHAPE), psf=psf, readout=WX, pattern=pattern, basis=basis, toeplitz=False
    )

    x = _rand(*fast.ishape)
    torch.testing.assert_close(fast(x), slow(x), rtol=0, atol=0)

    want = slow.adjoint(slow(x))
    assert (fast.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_collapsed_subspace_encoding_is_still_one_bart_operator(maps, basis, frame_pattern):
    """The normal is composed from operators, so both sides stay BART's."""
    A = linop.CartesianSense(maps, (COEFFS, Y, X), pattern=frame_pattern, basis=basis)
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
    return _rand(SETS, COILS, Y, X)


def test_espirit_hands_its_second_map_over_as_it_stands():
    """No reshaping between the calibration and the encoding.

    ``ecalib`` returns ``(sets, coils, 1, y, x)``, which is a bank over three
    spatial axes with one slice, so the encoding is the three-dimensional one
    over an image with a z of one.
    """
    import bartorch.tools as bt

    kspace = bt.phantom([Y, Y], coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=SETS)
    assert maps.shape == (SETS, COILS, 1, Y, Y)

    A = linop.CartesianSense(maps, (SETS, 1, Y, Y))
    assert A.ishape == (SETS, 1, Y, Y)
    assert A.oshape == (COILS, 1, Y, Y)


def test_a_two_map_reconstruction_fits_the_data_it_was_given():
    """Not the image it was given: two sets of ESPIRiT maps do not span an
    arbitrary pair of images, which is the whole point of the second one.  The
    operator has a null space, so what CG converges to is a least-squares
    solution -- and it is the data that has to come back, not the image."""
    import bartorch.tools as bt

    kspace = bt.phantom([Y, Y], coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=SETS)

    A = linop.CartesianSense(maps, (SETS, 1, Y, Y))
    torch.manual_seed(0)
    y = A(_rand(*A.ishape))
    got = CG(maxiter=80)(y, A)
    assert (A(got) - y).abs().max() / y.abs().max() < 1e-2


def test_a_cartesian_encoding_over_sets_is_the_sum_of_the_one_set_ones(bank):
    A = linop.CartesianSense(bank, (SETS, Y, X))
    torch.manual_seed(0)
    x = _rand(*A.ishape)

    want = sum(linop.CartesianSense(bank[m], (Y, X))(x[m]) for m in range(SETS))
    torch.testing.assert_close(A(x), want, rtol=1e-4, atol=1e-5)


def test_a_pattern_and_several_sets_go_together(bank):
    mask = (torch.rand(Y, 1) > 0.4).to(torch.complex64)
    A = linop.CartesianSense(bank, (SETS, Y, X), pattern=mask)
    assert _adjointness(A) < 1e-5


def test_a_subspace_and_several_sets_go_together(bank, basis, frame_pattern):
    A = linop.CartesianSense(bank, (SETS, COEFFS, Y, X), pattern=frame_pattern, basis=basis)
    slow = linop.CartesianSense(
        bank, (SETS, COEFFS, Y, X), pattern=frame_pattern, basis=basis, toeplitz=False
    )

    assert A.ishape == (SETS, COEFFS, Y, X)
    assert A.oshape == (COILS, FRAMES, Y, X)
    assert _adjointness(A) < 1e-5

    torch.manual_seed(0)
    x = _rand(*A.ishape)
    want = slow.adjoint(slow(x))
    assert (A.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_wave_carries_several_sets_too(wave_parts):
    _, psf, mask = wave_parts
    torch.manual_seed(10)
    sets_bank = _rand(SETS, COILS, Z, 5, SX)

    A = linop.WaveSense(sets_bank, (SETS, *SHAPE), psf=psf, readout=WX, pattern=mask)
    assert A.ishape == (SETS, Z, 5, SX)
    assert A.oshape == (COILS, Z, 5, WX)
    assert _adjointness(A) < 1e-5


# --- the native Cartesian operator --------------------------------------------
#
# With a pattern or a basis, CartesianSense is one operator in C: the pattern
# and the basis run inside the coil loop, and the normal transforms only the
# axes the pattern varies along.  Held against the model written out in torch.


def _fftc(t, axes):
    """The centred unitary transform, in torch."""
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(t, dim=axes), dim=axes, norm="ortho"), dim=axes
    )


def _ifftc(t, axes):
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(t, dim=axes), dim=axes, norm="ortho"), dim=axes
    )


def test_a_pattern_or_a_basis_builds_the_native_operator(maps, basis, frame_pattern):
    from bartorch.linop.mri import _CartesianNative

    mask = (torch.rand(Y, 1) > 0.5).to(torch.complex64)
    assert isinstance(linop.CartesianSense(maps, (Y, X), pattern=mask), _CartesianNative)
    assert isinstance(
        linop.CartesianSense(maps, (COEFFS, Y, X), pattern=frame_pattern, basis=basis),
        _CartesianNative,
    )


def test_the_native_normal_with_a_pattern_is_the_model_written_out():
    """A pattern of phase encodes, flat along the readout, on a 3D grid."""
    torch.manual_seed(6)
    coils, z, y, x = 3, 6, 8, 10
    maps = _rand(coils, z, y, x)
    mask = (torch.rand(z, y, 1) > 0.5).to(torch.complex64)
    A = linop.CartesianSense(maps, (z, y, x), pattern=mask)

    image = _rand(z, y, x)
    k = _fftc(maps * image, (-3, -2, -1)) * mask
    want = (maps.conj() * _ifftc(k * mask.conj(), (-3, -2, -1))).sum(0)

    got = A.normal(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_the_native_subspace_normal_is_the_model_written_out():
    """Frames, a basis and their pattern, collapsed into one kernel."""
    torch.manual_seed(7)
    coils, z, y, x, frames, coeffs = 3, 4, 8, 6, 5, 2
    maps = _rand(coils, z, y, x)
    b = _rand(coeffs, frames)
    pattern = (torch.rand(frames, z, y, 1) > 0.5).to(torch.complex64)
    A = linop.CartesianSense(maps, (coeffs, z, y, x), pattern=pattern, basis=b)

    image = _rand(*A.ishape)
    coeff_images = image.reshape(coeffs, 1, z, y, x)
    k = _fftc(maps[None] * coeff_images, (-3, -2, -1))
    frame_k = torch.einsum("kt,kczyx->tczyx", b, k) * pattern[:, None]
    back = torch.einsum("kt,tczyx->kczyx", b.conj(), frame_k * pattern[:, None].conj())
    want = (maps.conj()[None] * _ifftc(back, (-3, -2, -1))).sum(1).reshape(A.ishape)

    got = A.normal(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


# On a card the normal runs inside cuFFT's transforms, so what is checked there
# is that it did, and that it is still the model written out.

import bartorch  # noqa: E402  (for the card marker)
from bartorch._lib import library  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not bartorch._cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 16, 12), (6, 8, 10)])
def test_on_a_card_the_normal_runs_through_cufft_and_is_the_model_written_out(shape):
    """Host arrays and the operator on a card, with a pattern that varies along
    one axis: cuFFT links no callbacks into a transform along one axis, so the
    normal transforms a second, flat axis as well and is the same normal."""
    torch.manual_seed(8)
    coils = 3
    z, y, x = shape
    maps = _rand(coils, z, y, x)
    mask = (torch.rand(1, y, 1) > 0.5).to(torch.complex64)
    A = linop.CartesianSense(maps, (z, y, x), pattern=mask, device="cuda")

    image = _rand(z, y, x)
    before = library().bartorch_grid_fused()
    got = A.normal(image)
    assert library().bartorch_grid_fused() > before, "the normal ran through cuFFT's callbacks"

    k = _fftc(maps * image, (-3, -2, -1)) * mask
    want = (maps.conj() * _ifftc(k * mask.conj(), (-3, -2, -1))).sum(0)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


@requires_cuda
def test_on_a_card_the_subspace_normal_runs_through_cufft_and_is_the_model_written_out():
    torch.manual_seed(9)
    coils, z, y, x, frames, coeffs = 3, 4, 8, 6, 5, 2
    maps = _rand(coils, z, y, x)
    b = _rand(coeffs, frames)
    pattern = (torch.rand(frames, z, y, 1) > 0.5).to(torch.complex64)
    A = linop.CartesianSense(maps, (coeffs, z, y, x), pattern=pattern, basis=b, device="cuda")

    image = _rand(*A.ishape)
    before = library().bartorch_grid_fused()
    got = A.normal(image)
    assert library().bartorch_grid_fused() > before, "the normal ran through cuFFT's callbacks"

    coeff_images = image.reshape(coeffs, 1, z, y, x)
    k = _fftc(maps[None] * coeff_images, (-3, -2, -1))
    frame_k = torch.einsum("kt,kczyx->tczyx", b, k) * pattern[:, None]
    back = torch.einsum("kt,tczyx->kczyx", b.conj(), frame_k * pattern[:, None].conj())
    want = (maps.conj()[None] * _ifftc(back, (-3, -2, -1))).sum(1).reshape(A.ishape)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


# --- sampled-only k-space ----------------------------------------------------
#
# A table of phase encodes with the whole readout along each, checked against
# the dense k-space of the model written out, read at the same places.


def _read_table(k, positions):
    """Dense k-space `k` of (coils, [frames,] [z,] y, x) at `positions` of
    ([frames,] shots, d): (coils, [frames,] shots, x), zero at padding."""
    frames = positions.shape[:-2]
    k = k.reshape(k.shape[0], -1, *k.shape[-positions.shape[-1] - 1 :])
    pos = positions.reshape(-1, *positions.shape[-2:])
    out = torch.zeros(k.shape[0], pos.shape[0], pos.shape[1], k.shape[-1], dtype=k.dtype)
    for t in range(pos.shape[0]):
        for s in range(pos.shape[1]):
            index = tuple(int(i) for i in pos[t, s])
            if min(index) >= 0:
                out[:, t, s] = k[(slice(None), t, *index)]
    return out.reshape(k.shape[0], *frames, pos.shape[1], k.shape[-1])


def _counts_pattern(positions, plane):
    """The dense pattern with the same normal: the square root of how often a
    place is sampled, per frame, flat along the readout."""
    pos = positions.reshape(-1, *positions.shape[-2:])
    counts = torch.zeros(pos.shape[0], *plane, dtype=torch.float64)
    for t in range(pos.shape[0]):
        for s in range(pos.shape[1]):
            index = tuple(int(i) for i in pos[t, s])
            if min(index) >= 0:
                counts[(t, *index)] += 1
    return counts.sqrt().to(torch.complex64).reshape(*positions.shape[:-2], *plane, 1)


@pytest.fixture
def positions_2d():
    """Nine phase encodes of Y, one of them twice, and a padding entry."""
    torch.manual_seed(5)
    pos = torch.randperm(Y)[:9].reshape(9, 1)
    return torch.cat([pos, pos[:1], torch.full((1, 1), -1)])


@pytest.mark.parametrize("readout", ["kspace", "image"])
def test_sampled_samples_are_the_dense_ones_at_the_positions(maps, positions_2d, readout):
    A = linop.CartesianSense(maps, (Y, X), positions=positions_2d, readout=readout)
    assert A.oshape == (COILS, positions_2d.shape[0], X)

    image = _rand(Y, X)
    axes = (-2, -1) if readout == "kspace" else (-2,)
    want = _read_table(_fftc(maps * image, axes), positions_2d)
    got = A(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5
    assert got[:, -1].abs().max() == 0


@pytest.mark.parametrize("readout", ["kspace", "image"])
def test_sampled_encoding_has_the_adjoint_it_claims(maps, positions_2d, readout):
    A = linop.CartesianSense(maps, (Y, X), positions=positions_2d, readout=readout)
    assert _adjointness(A) < 1e-5


def test_sampled_normal_is_the_dense_one_with_the_same_counts(maps, positions_2d):
    """A place sampled twice counts twice, as the two applications count it."""
    fast = linop.CartesianSense(maps, (Y, X), positions=positions_2d)
    slow = linop.CartesianSense(maps, (Y, X), positions=positions_2d, toeplitz=False)
    dense = linop.CartesianSense(maps, (Y, X), pattern=_counts_pattern(positions_2d, (Y,)))

    image = _rand(Y, X)
    want = slow.adjoint(slow(image))
    for got in (fast.normal(image), dense.normal(image)):
        assert (got - want).abs().max() / want.abs().max() < 1e-5


def _positions_3d(frames, shots, z, y, seed):
    torch.manual_seed(seed)
    out = torch.full((frames, shots, 2), -1, dtype=torch.long)
    for t in range(frames):
        n = shots - (t % 2)  # frames of different lengths, padded
        flat = torch.randperm(z * y)[:n]
        out[t, :n, 0], out[t, :n, 1] = flat // y, flat % y
    return out


@pytest.mark.parametrize("readout", ["kspace", "image"])
def test_sampled_subspace_samples_are_the_dense_frames_at_the_positions(readout):
    torch.manual_seed(11)
    coils, z, y, x, frames, coeffs, shots = 3, 4, 6, 5, 5, 2, 7
    maps = _rand(coils, z, y, x)
    b = _rand(coeffs, frames)
    positions = _positions_3d(frames, shots, z, y, 12)
    A = linop.CartesianSense(maps, (coeffs, z, y, x), positions=positions, basis=b, readout=readout)
    assert A.oshape == (coils, frames, shots, x)

    image = _rand(coeffs, z, y, x)
    axes = (-3, -2, -1) if readout == "kspace" else (-3, -2)
    k = _fftc(maps[None] * image[:, None], axes)
    frame_k = torch.einsum("kt,kczyx->ctzyx", b, k)
    want = _read_table(frame_k, positions)
    got = A(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5
    assert _adjointness(A) < 1e-5


def test_sampled_subspace_normal_is_the_two_applications_and_the_dense_one():
    torch.manual_seed(13)
    coils, z, y, x, frames, coeffs, shots = 3, 4, 6, 5, 5, 2, 7
    maps = _rand(coils, z, y, x)
    b = _rand(coeffs, frames)
    positions = _positions_3d(frames, shots, z, y, 14)
    fast = linop.CartesianSense(maps, (coeffs, z, y, x), positions=positions, basis=b)
    slow = linop.CartesianSense(
        maps, (coeffs, z, y, x), positions=positions, basis=b, toeplitz=False
    )
    dense = linop.CartesianSense(
        maps, (coeffs, z, y, x), pattern=_counts_pattern(positions, (z, y)), basis=b
    )

    image = _rand(coeffs, z, y, x)
    want = slow.adjoint(slow(image))
    for got in (fast.normal(image), dense.normal(image)):
        assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_sampled_positions_are_checked(maps, positions_2d):
    with pytest.raises(ValueError, match="pattern or positions"):
        linop.CartesianSense(maps, (Y, X), pattern=torch.ones(Y, 1), positions=positions_2d)
    with pytest.raises(ValueError, match="indices"):
        linop.CartesianSense(maps, (Y, X), positions=torch.zeros(4, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        linop.CartesianSense(maps, (Y, X), positions=torch.full((4, 1), Y))
    with pytest.raises(ValueError, match="basis"):
        linop.CartesianSense(maps, (Y, X), positions=torch.zeros(3, 4, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="readout"):
        linop.CartesianSense(maps, (Y, X), positions=positions_2d, readout="hybrid")


@requires_cuda
@pytest.mark.parametrize("readout", ["kspace", "image"])
def test_on_a_card_sampled_samples_run_through_cufft_and_are_the_dense_ones(readout):
    torch.manual_seed(15)
    coils, z, y, x, frames, coeffs, shots = 3, 4, 6, 5, 5, 2, 7
    maps = _rand(coils, z, y, x)
    b = _rand(coeffs, frames)
    positions = _positions_3d(frames, shots, z, y, 16)
    A = linop.CartesianSense(
        maps, (coeffs, z, y, x), positions=positions, basis=b, readout=readout, device="cuda"
    )
    host = linop.CartesianSense(
        maps, (coeffs, z, y, x), positions=positions, basis=b, readout=readout
    )

    image = _rand(coeffs, z, y, x)
    before = library().bartorch_grid_fused()
    got = A(image)
    assert library().bartorch_grid_fused() > before, "the forward ran through cuFFT's callbacks"
    want = host(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5

    samples = _rand(*A.oshape)
    before = library().bartorch_grid_fused()
    got = A.adjoint(samples)
    assert library().bartorch_grid_fused() > before, "the adjoint ran through cuFFT's callbacks"
    want = host.adjoint(samples)
    assert (got - want).abs().max() / want.abs().max() < 1e-5

    want = host.normal(image)
    got = A.normal(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


@requires_cuda
def test_on_a_card_a_2d_sampled_encoding_is_the_host_one(maps, positions_2d):
    """In 2D the transform along the phase encode alone takes the readout with
    it (cuFFT links no callbacks into one axis), so the table is read in
    k-space along the readout and transformed back for `image`."""
    for readout in ("kspace", "image"):
        A = linop.CartesianSense(
            maps, (Y, X), positions=positions_2d, readout=readout, device="cuda"
        )
        host = linop.CartesianSense(maps, (Y, X), positions=positions_2d, readout=readout)
        image = _rand(Y, X)
        want = host(image)
        assert (A(image) - want).abs().max() / want.abs().max() < 1e-5
        samples = _rand(*A.oshape)
        want = host.adjoint(samples)
        assert (A.adjoint(samples) - want).abs().max() / want.abs().max() < 1e-5


# --- wave in the coil loop ---------------------------------------------------
#
# The table of a wave encoding is the dense one read at the positions, and its
# normal is the dense one over the pattern with the same counts.


@pytest.fixture
def wave_subspace():
    torch.manual_seed(21)
    coils, z, y, x, wx, frames, coeffs, shots = 3, 4, 6, 5, 10, 5, 2, 7
    maps = _rand(coils, z, y, x)
    psf = torch.exp(1j * torch.randn(z, y, wx, dtype=torch.float64)).to(torch.complex64)
    b = _rand(coeffs, frames)
    positions = _positions_3d(frames, shots, z, y, 22)
    return maps, psf, b, positions, (coeffs, z, y, x), wx


@pytest.mark.parametrize("centred", [False, True])
def test_sampled_wave_samples_are_the_dense_ones_at_the_positions(wave_subspace, centred):
    maps, psf, b, positions, shape, wx = wave_subspace
    A = linop.WaveSense(
        maps, shape, psf=psf, readout=wx, positions=positions, basis=b, centred=centred
    )
    dense = linop.WaveSense(maps, shape, psf=psf, readout=wx, basis=b, centred=centred)
    assert A.oshape == (maps.shape[0], b.shape[1], positions.shape[1], wx)

    image = _rand(*shape)
    want = _read_table(dense(image), positions)
    got = A(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5
    assert _adjointness(A) < 1e-5


@pytest.mark.parametrize("centred", [False, True])
def test_sampled_wave_normal_is_the_two_applications_and_the_dense_one(wave_subspace, centred):
    maps, psf, b, positions, shape, wx = wave_subspace
    common = dict(readout=wx, basis=b, centred=centred)
    fast = linop.WaveSense(maps, shape, psf=psf, positions=positions, **common)
    slow = linop.WaveSense(maps, shape, psf=psf, positions=positions, toeplitz=False, **common)
    dense = linop.WaveSense(
        maps, shape, psf=psf, pattern=_counts_pattern(positions, shape[1:3]), **common
    )

    image = _rand(*shape)
    want = slow.adjoint(slow(image))
    for got in (fast.normal(image), dense.normal(image)):
        assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_a_wave_pattern_along_the_readout_keeps_the_two_applications(wave_parts):
    """The readout is not the phase-encode transform's to cancel."""
    maps, psf, mask = wave_parts
    A = linop.WaveSense(maps, SHAPE, psf=psf, readout=WX, pattern=mask)
    x = _rand(*A.ishape)
    want = A.adjoint(A(x))
    got = A.normal(x)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


@requires_cuda
@pytest.mark.parametrize("centred", [False, True])
def test_on_a_card_the_wave_normal_runs_through_cufft_and_is_the_host_one(wave_subspace, centred):
    maps, psf, b, positions, shape, wx = wave_subspace
    pattern = _counts_pattern(positions, shape[1:3])
    common = dict(readout=wx, pattern=pattern, basis=b, centred=centred)
    A = linop.WaveSense(maps, shape, psf=psf, device="cuda", **common)
    host = linop.WaveSense(maps, shape, psf=psf, **common)

    image = _rand(*shape)
    before = library().bartorch_grid_fused()
    got = A.normal(image)
    assert library().bartorch_grid_fused() > before, "the normal ran through cuFFT's callbacks"
    want = host.normal(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5

    want = host(image)
    assert (A(image) - want).abs().max() / want.abs().max() < 1e-5
    samples = _rand(*A.oshape)
    want = host.adjoint(samples)
    assert (A.adjoint(samples) - want).abs().max() / want.abs().max() < 1e-5


@requires_cuda
@pytest.mark.parametrize("centred", [False, True])
def test_on_a_card_a_sampled_wave_runs_through_cufft_and_is_the_host_one(wave_subspace, centred):
    maps, psf, b, positions, shape, wx = wave_subspace
    common = dict(readout=wx, positions=positions, basis=b, centred=centred)
    A = linop.WaveSense(maps, shape, psf=psf, device="cuda", **common)
    host = linop.WaveSense(maps, shape, psf=psf, **common)

    image = _rand(*shape)
    before = library().bartorch_grid_fused()
    got = A(image)
    assert library().bartorch_grid_fused() > before, "the forward ran through cuFFT's callbacks"
    want = host(image)
    assert (got - want).abs().max() / want.abs().max() < 1e-5

    samples = _rand(*A.oshape)
    want = host.adjoint(samples)
    assert (A.adjoint(samples) - want).abs().max() / want.abs().max() < 1e-5

    want = host.normal(image)
    assert (A.normal(image) - want).abs().max() / want.abs().max() < 1e-5


# --- the wave point-spread function from the gradient wave ------------------
#
# Checked against BART's own ``wavepsf``, which makes the sine wave along y
# with ``-c`` making the cosine one, combined as its help says with ``fmac``.

WAVE = dict(max_grad=0.8, max_slew=17000.0, cycles=6, adc=3.0)
TOOL = dict(a=3000, t=1e-5, g=0.8, s=17000.0, n=6)


def test_the_wave_psf_along_y_is_barts_wavepsf():
    import bartorch.tools as bt
    from bartorch.linop.mri import _wave_psf

    wx, ny, dy = 64, 16, 0.1
    got = _wave_psf(wx, (ny,), resolution=dy, offset=0.0, **WAVE)
    want = bt.wavepsf(x=wx, y=ny, r=dy, **TOOL).reshape(ny, wx)
    torch.testing.assert_close(got.to(want.dtype), want, rtol=1e-4, atol=1e-4)


def test_the_wave_psf_in_3d_is_barts_cosine_wave_along_z_times_the_sine_along_y():
    import bartorch.tools as bt
    from bartorch.linop.mri import _wave_psf

    wx, nz, ny, dz, dy = 64, 8, 16, 0.2, 0.1
    got = _wave_psf(wx, (nz, ny), resolution=(dz, dy), offset=0.0, **WAVE)
    along_z = bt.wavepsf(c=True, x=wx, y=nz, r=dz, **TOOL).reshape(nz, 1, wx)
    along_y = bt.wavepsf(x=wx, y=ny, r=dy, **TOOL).reshape(1, ny, wx)
    want = along_z * along_y
    torch.testing.assert_close(got.to(want.dtype), want, rtol=1e-4, atol=1e-4)


def test_an_offset_moves_the_wave_psf_along_its_axis():
    """An isocentre two voxels away is the same function two voxels along."""
    from bartorch.linop.mri import _wave_psf

    wx, ny, dy = 64, 16, 0.1
    base = _wave_psf(wx, (ny,), resolution=dy, offset=0.0, **WAVE)
    moved = _wave_psf(wx, (ny,), resolution=dy, offset=2 * dy, **WAVE)
    torch.testing.assert_close(moved[2:], base[:-2])


def test_wave_sense_makes_its_psf_from_the_gradient_wave(wave_parts):
    from bartorch.linop.mri import _wave_psf

    maps, _, mask = wave_parts
    made = linop.WaveSense(maps, SHAPE, readout=WX, pattern=mask, resolution=(0.2, 0.1), **WAVE)
    psf = _wave_psf(WX, SHAPE[:-1], resolution=(0.2, 0.1), offset=0.0, **WAVE)
    given = linop.WaveSense(maps, SHAPE, readout=WX, pattern=mask, psf=psf)

    x = _rand(*given.ishape)
    torch.testing.assert_close(made(x), given(x), rtol=1e-6, atol=1e-6)


def test_wave_sense_says_what_its_psf_is_missing(wave_parts):
    maps, psf, _ = wave_parts
    with pytest.raises(ValueError, match="cycles, adc missing"):
        linop.WaveSense(maps, SHAPE, readout=WX, max_grad=0.8, max_slew=17000.0, resolution=0.1)
    with pytest.raises(ValueError, match="one or the other"):
        linop.WaveSense(maps, SHAPE, readout=WX, psf=psf, cycles=6)


# --- time segmentation over a NUFFT -------------------------------------------
#
# Written out, the encoding is the sum of the segments; over a non-Cartesian
# SENSE operator it is one subspace operator whose basis lies along the samples.


@pytest.fixture
def segmented():
    import bartorch.tools as bt

    torch.manual_seed(31)
    n, coils, segments = 16, 3, 3
    maps = _rand(coils, n, n)
    traj = bt.traj(x=n, y=24, r=True)
    t = torch.arange(n, dtype=torch.float64) / n
    b = torch.stack([torch.exp(-2j * torch.pi * (s + 1) * t) for s in range(segments)])
    b = b.to(torch.complex64).reshape(segments, 1, n)
    c = (torch.randn(segments, n, n, dtype=torch.complex64) * 0.1 + 1.0) / segments
    return maps, traj, b, c, n


def test_a_segmented_nufft_is_the_sum_of_its_segments(segmented):
    maps, traj, b, c, n = segmented
    from bartorch.linop.base import _WithNormal

    E = linop.NoncartesianSense(maps, (n, n), traj=traj)
    A = linop.FieldCorrected(E, coefficients=(b, c))
    assert isinstance(A, _WithNormal), "the segments are one subspace operator, not their sum"
    assert (A.ishape, A.oshape) == (E.ishape, E.oshape)

    x = _rand(n, n)
    want = sum(b[s] * E(c[s] * x) for s in range(b.shape[0]))
    got = A(x)
    assert (got - want).abs().max() / want.abs().max() < 1e-5

    y = _rand(*A.oshape)
    want = sum(c[s].conj() * E.adjoint(b[s].conj() * y) for s in range(b.shape[0]))
    got = A.adjoint(y)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_a_segmented_nufft_normal_is_the_two_applications(segmented):
    """The Toeplitz normal, to within the transform's own tolerance."""
    maps, traj, b, c, n = segmented
    E = linop.NoncartesianSense(maps, (n, n), traj=traj)
    A = linop.FieldCorrected(E, coefficients=(b, c))

    x = _rand(n, n)
    want = A.adjoint(A(x))
    got = A.normal(x)
    assert (got - want).abs().max() / want.abs().max() < 1e-2


@requires_cuda
def test_on_a_card_a_segmented_nufft_is_the_host_one(segmented):
    maps, traj, b, c, n = segmented
    A = linop.FieldCorrected(
        linop.NoncartesianSense(maps, (n, n), traj=traj, device="cuda"), coefficients=(b, c)
    )
    host = linop.FieldCorrected(
        linop.NoncartesianSense(maps, (n, n), traj=traj), coefficients=(b, c)
    )

    from bartorch.linop.base import _WithNormal

    assert isinstance(A, _WithNormal) and isinstance(host, _WithNormal)

    # cuFINUFFT and FINUFFT agree on a plain NUFFT to a few parts in 1e3.
    x = _rand(n, n)
    want = host(x)
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-2
    y = _rand(*A.oshape)
    want = host.adjoint(y)
    assert (A.adjoint(y) - want).abs().max() / want.abs().max() < 1e-2
    want = host.adjoint(host(x))
    assert (A.normal(x) - want).abs().max() / want.abs().max() < 1e-2
