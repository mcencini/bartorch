"""The FINUFFT-backed NUFFT, as an operator and underneath BART's own tools,
against an explicit discrete Fourier sum and against BART's own gridder.
"""

import logging
import os

import numpy as np
import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _finufft
from bartorch.ops import LinearOperator

requires_finufft = pytest.mark.skipif(not _finufft.available(), reason="finufft is not installed")


def _within_tolerance(got, ref):
    """As close to the explicit sum as the tolerance the plan was made with.

    FINUFFT delivers the tolerance it is given and no more, so a bound tied to
    it is what the transform promises.  What these tests are really pinning is
    the sign, the scaling and the axis order, and a wrong one of those misses
    by orders of magnitude rather than by a factor of a few.
    """
    error = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert error < 5 * _finufft.tolerance(), (error, _finufft.tolerance())


def _all_finufft(least=1):
    """FINUFFT built every operator, and BART's gridder built none.

    A tool with a Toeplitz normal builds more than one: the transform pair the
    caller asked for, and the one transform a point spread function is made
    from.  What matters is that none of them was BART's.
    """
    built, bart = _finufft.operators_built()
    assert bart == 0, _finufft.decline_reason()
    assert built >= least, (built, least)


def _sides_agree(card, host):
    """Two libraries promised the same tolerance agree to about it.

    Element-wise closeness is the wrong measure where a transform passes
    through zero, so this is the difference over the norm of what it is held
    against.
    """
    error = (card - host).norm().item() / host.norm().item()
    assert error < 5 * _finufft.tolerance(), (error, _finufft.tolerance())


def _inner(a, b):
    return torch.vdot(a.flatten(), b.flatten()).real.item()


@requires_finufft
def test_matches_an_explicit_dft_on_a_radial_trajectory():
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    y = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)(img)

    trj = traj.numpy().real
    im = img.numpy().reshape(n, n)
    kx, ky = trj[..., 0], trj[..., 1]
    x = np.arange(n) - n // 2
    phase = np.exp(
        -2j
        * np.pi
        * (
            kx[..., None, None] * x[None, None, None, :] / n
            + ky[..., None, None] * x[None, None, :, None] / n
        )
    )
    ref = (phase * im[None, None]).sum(axis=(-1, -2)) / n
    assert np.linalg.norm(y.numpy().reshape(ref.shape) - ref) / np.linalg.norm(ref) < 5e-3


@requires_finufft
def test_agrees_with_barts_own_gridder():
    """The substitution and the thing it replaces compute the same operator."""
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)

    fast = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    with _finufft.barts_own_gridder():
        bart = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    assert fast.ishape == bart.ishape and fast.oshape == bart.oshape
    a, b = bart(img), fast(img)
    assert (a - b).norm().item() / a.norm().item() < 5e-3


@requires_finufft
def test_adjoint_identity_holds():
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    x = torch.randn(*A.ishape, dtype=torch.complex64)
    y = torch.randn(*A.oshape, dtype=torch.complex64)
    assert _inner(A(x), y) == pytest.approx(_inner(x, A.adjoint(y)), rel=1e-3)


@requires_finufft
def test_it_carries_coils_through_one_plan():
    # BART puts the coils past the three spatial axes, so a two-dimensional
    # coil image is (coils, 1, y, x) and its k-space (coils, spokes, readout, 1).
    n, ncoils, spokes = 32, 4, 16
    traj = bt.traj(x=n, y=spokes, r=True)
    A = LinearOperator.nufft(traj, (ncoils, 1, n, n), (ncoils, spokes, n, 1), toeplitz=False)
    assert A.oshape[0] == ncoils
    x = torch.randn(ncoils, 1, n, n, dtype=torch.complex64)
    y = A(x)
    # Each coil must transform independently of the others.
    single = LinearOperator.nufft(traj, (1, 1, n, n), (1, spokes, n, 1), toeplitz=False)
    for c in range(ncoils):
        torch.testing.assert_close(
            y[c].reshape(-1), single(x[c : c + 1]).reshape(-1), rtol=1e-4, atol=1e-4
        )


@requires_finufft
def test_bart_solves_against_a_finufft_operator():
    # The point of matching BART's convention: the operator goes into BART's
    # own conjugate gradients unchanged.
    n = 32
    traj = bt.traj(x=n, y=64, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    y = A(img)
    x = A.lstsq(y, lambda_=1e-3, maxiter=30)
    assert x.shape == img.shape
    assert (x - img).norm().item() / img.norm().item() < 0.5


@pytest.fixture
def in_tools():
    """BART's tools computing their NUFFT with FINUFFT, for one test.

    That is where the library starts, so what this restores afterwards is the
    substitution rather than BART's gridder: a test that turns it off must not
    leave the next one quietly on it.
    """
    if not _finufft.use_in_tools():
        pytest.skip("the substitution declined to install itself")
    yield
    _finufft.use_in_tools(True)


def _dft(traj, image, n):
    """The transform BART's NUFFT computes, summed out one sample at a time."""
    trj = traj.numpy().real
    im = image.numpy().reshape(n, n)
    kx, ky = trj[..., 0], trj[..., 1]
    x = np.arange(n) - n // 2
    phase = np.exp(
        -2j
        * np.pi
        * (
            kx[..., None, None] * x[None, None, None, :] / n
            + ky[..., None, None] * x[None, None, :, None] / n
        )
    )
    return (phase * im[None, None]).sum(axis=(-1, -2)) / n


@requires_finufft
def test_barts_nufft_tool_matches_an_explicit_dft_with_finufft_underneath(in_tools):
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)

    _finufft.reset_counters()
    y = bt.nufft(traj, img)

    _all_finufft()
    ref = _dft(traj, img, n)
    _within_tolerance(y.numpy().reshape(ref.shape), ref)


@requires_finufft
def test_the_substituted_operator_is_its_own_adjoint_pair(in_tools):
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    x = torch.randn(1, n, n, dtype=torch.complex64)
    y = torch.randn(16, n, 1, dtype=torch.complex64)

    ax = bt.nufft(traj, x)
    ahy = bt.nufft(traj, y, adjoint=True, image_dims=(n, n, 1))

    assert _inner(ax, y) == pytest.approx(_inner(x, ahy), rel=1e-4)


@requires_finufft
def test_weights_multiply_the_transform_and_their_conjugate_its_adjoint(in_tools):
    n, spokes = 32, 16
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    weights = torch.rand(spokes, n, 1).to(torch.complex64)

    _finufft.reset_counters()
    y = bt.nufft(traj, img, p=weights)

    _all_finufft()
    ref = _dft(traj, img, n) * weights.numpy().reshape(spokes, n)
    _within_tolerance(y.numpy().reshape(ref.shape), ref)

    x = torch.randn(1, n, n, dtype=torch.complex64)
    k = torch.randn(spokes, n, 1, dtype=torch.complex64)
    adjoint = bt.nufft(traj, k, adjoint=True, image_dims=(n, n, 1), p=weights)
    assert _inner(bt.nufft(traj, x, p=weights), k) == pytest.approx(_inner(x, adjoint), rel=1e-4)


@requires_finufft
def test_pics_reconstructs_the_same_image_either_way(in_tools):
    n = 64
    traj = bt.traj(x=n, y=128, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    kspace = bt.nufft(traj, img)
    maps = torch.ones(1, n, n, dtype=torch.complex64)

    _finufft.reset_counters()
    fast = bt.pics(kspace, maps, t=traj)
    _all_finufft()

    with _finufft.barts_own_gridder():
        _finufft.reset_counters()
        reference = bt.pics(kspace, maps, t=traj)
        assert _finufft.operators_built() == (0, 1)

    assert (fast - reference).norm().item() / reference.norm().item() < 0.05


@requires_finufft
def test_the_normal_solves_the_normal_equations(in_tools):
    # A^H A is a convolution, so the substituted operator answers it with
    # BART's point spread function rather than a transform each way.  What
    # pins it is the solve: an iterative inverse driven by a wrong normal
    # lands somewhere else.
    n, lam = 16, 1e-2
    traj = bt.traj(x=n, y=32, r=True)
    image = bt.phantom([n, n]).reshape(1, n, n)
    kspace = bt.nufft(traj, image)

    _finufft.reset_counters()
    got = bt.nufft(traj, kspace, inverse=True, image_dims=(n, n, 1), l2_reg=lam, max_iter=200)
    assert _finufft.normals_built() == (1, 0), "the solve did not run on a point spread function"

    trj = traj.numpy().real
    x = np.arange(n) - n // 2
    E = (
        np.exp(
            -2j
            * np.pi
            * (
                trj[..., 0][..., None, None] * x[None, None, None, :] / n
                + trj[..., 1][..., None, None] * x[None, None, :, None] / n
            )
        ).reshape(-1, n * n)
        / n
    )
    y = kspace.numpy().reshape(-1)
    ref = np.linalg.solve(E.conj().T @ E + lam * np.eye(n * n), E.conj().T @ y).reshape(n, n)

    err = np.linalg.norm(got.numpy().reshape(n, n) - ref) / np.linalg.norm(ref)
    assert err < 5e-2, f"the solve landed {err:.2e} from the explicit one"


@requires_finufft
def test_the_two_normals_solve_the_same_problem(in_tools):
    n = 64
    traj = bt.traj(x=n, y=128, r=True)
    image = bt.phantom([n, n]).reshape(1, n, n)
    kspace = bt.nufft(traj, image)
    maps = torch.ones(1, n, n, dtype=torch.complex64)

    _finufft.reset_counters()
    fast = bt.pics(kspace, maps, t=traj)
    assert _finufft.normals_built() == (1, 0)

    _finufft.reset_counters()
    pair = bt.pics(kspace, maps, t=traj, no_toeplitz=True)
    assert _finufft.normals_built() == (0, 1)

    assert (fast - pair).norm().item() / pair.norm().item() < 0.05


@requires_finufft
def test_barts_own_gridder_is_reachable_only_from_inside_the_package():
    """It is there for holding the substitution against, and for nothing else."""
    import bartorch

    assert not hasattr(bartorch.finufft, "disable")
    assert not hasattr(bartorch.finufft, "enable")

    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)

    with _finufft.barts_own_gridder():
        _finufft.reset_counters()
        bt.nufft(traj, img)
        assert _finufft.operators_built() == (0, 1)

    _finufft.reset_counters()
    bt.nufft(traj, img)
    assert _finufft.operators_built() == (1, 0), "the block put it back"
    assert not _finufft.fallback_allowed()


@requires_finufft
def test_the_operator_is_unaffected_by_the_substitution():
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)

    _finufft.use_in_tools(False)
    y = A(img)
    _finufft.use_in_tools()
    try:
        torch.testing.assert_close(A(img), y, rtol=1e-5, atol=1e-5)
    finally:
        _finufft.use_in_tools(False)


@requires_finufft
def test_the_device_transform_is_offered_only_where_cufinufft_is(in_tools):
    # The two libraries are registered together and picked by where the data
    # is, so a trajectory on a card is BART's own operator's without the
    # cufinufft wheel rather than a transform that quietly runs on the host.
    assert _finufft.used_on_device() == (_finufft.cuda_available() and bartorch.cuda.built()), (
        _finufft.decline_reason()
    )


@requires_finufft
@pytest.mark.skipif(
    not (bartorch.cuda.available() and _finufft.cuda_available()),
    reason="this needs a CUDA device and the cufinufft package",
)
def test_a_trajectory_on_a_card_is_transformed_by_cufinufft(in_tools):
    n = 32
    traj = bt.traj(x=n, y=16, r=True).cuda()
    image = bt.phantom([n, n]).reshape(1, n, n).cuda()

    _finufft.reset_counters()
    y = bt.nufft(traj, image)

    assert y.device.type == "cuda"
    _all_finufft()
    ref = _dft(traj.cpu(), image.cpu(), n)
    _within_tolerance(y.cpu().numpy().reshape(ref.shape), ref)


@requires_finufft
@pytest.mark.skipif(
    not (bartorch.cuda.available() and _finufft.cuda_available()),
    reason="this needs a CUDA device and the cufinufft package",
)
def test_one_operator_serves_both_sides_of_the_bus(in_tools):
    """BART applies one operator to memory on either side, so it plans on both.

    ``pics`` takes its first adjoint from the k-space it mapped and then
    iterates on device vectors; a plan belongs to the library that made it, so
    the operator has to answer both with the same numbers.
    """
    n = 32
    traj = bt.traj(x=n, y=16, r=True).cuda()
    image = bt.phantom([n, n]).reshape(1, n, n)

    _finufft.reset_counters()
    A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    _all_finufft()

    on_card = A(image.cuda())
    on_host = A(image)
    assert on_card.device.type == "cuda" and on_host.device.type == "cpu"

    ref = _dft(traj.cpu(), image, n)
    got = on_card.cpu().numpy().reshape(ref.shape)
    _within_tolerance(got, ref)
    _sides_agree(on_card.cpu(), on_host)


@requires_finufft
def test_more_frames_than_a_batch_of_one_thousand_are_still_finuffts(in_tools):
    """FINUFFT batches against one point set, so frame count is not what decides.

    A dynamic dataset carries far more frames than coils, and each frame is
    another transform over the same trajectory.  FINUFFT takes them as one
    plan and slices them internally, so the operator has no reason to hand a
    long series back to BART.
    """
    n, frames = 16, 1500
    traj = bt.traj(x=n, y=8, r=True)
    # Frames sit beyond the three spatial axes, which is what makes them a batch.
    image = torch.zeros(frames, 1, n, n, dtype=torch.complex64)
    image[..., n // 2, n // 2] = 1.0  # a point source at the centre of every frame

    _finufft.reset_counters()
    A = LinearOperator.nufft(
        traj, (frames, 1, n, n), kspace_shape=(frames, 8, n, 1), toeplitz=False
    )
    _all_finufft()

    y = A(image)
    assert y.shape[0] == frames

    # Every frame holds the same source, so every frame holds the same samples.
    torch.testing.assert_close(y[0], y[-1])
    ref = _dft(traj, image[:1], n)
    got = y[0].numpy().reshape(ref.shape)
    _within_tolerance(got, ref)


def _subspace(n, spokes, frames, coeffs):
    """A trajectory that varies across frames, and a basis over them.

    BART puts frames on TE and coefficients on COEFF, so in C order the
    trajectory is ``(frames, 1, 1, spokes, readout, 3)`` and the basis
    ``(coeffs, frames, 1, 1, 1, 1, 1)``.
    """
    traj = bt.traj(x=n, y=spokes * frames, r=True).reshape(frames, spokes, n, 3)[:, None, None]
    basis = torch.zeros(coeffs, frames, 1, 1, 1, 1, 1, dtype=torch.complex64)
    basis[0, :, 0, 0, 0, 0, 0] = 1.0
    basis[1, :, 0, 0, 0, 0, 0] = torch.linspace(-1, 1, frames)
    return traj, basis


def _phase_per_frame(traj, n, sign):
    """``exp(sign * 2i pi k.x / n)`` for every sample of every frame."""
    frames, spokes = traj.shape[0], traj.shape[3]
    k = traj.numpy().real.reshape(frames, spokes, n, 3)
    g = np.arange(n) - n // 2
    return np.exp(
        sign
        * 2j
        * np.pi
        * (
            k[..., 0][..., None, None] * g[None, None, None, None, :] / n
            + k[..., 1][..., None, None] * g[None, None, None, :, None] / n
        )
    )


@requires_finufft
def test_a_subspace_adjoint_over_a_per_frame_trajectory_matches_an_explicit_sum(in_tools):
    """One plan over the whole raveled trajectory serves every coefficient.

    Each frame is acquired along its own trajectory, and the adjoint sums all
    of them into one image per coefficient, weighted by the conjugate basis.
    The samples of every coefficient are the same points, so the operator
    plans once and the basis is a contraction either side of the transform.
    """
    n, spokes, frames, coeffs = 16, 5, 4, 2
    traj, basis = _subspace(n, spokes, frames, coeffs)
    torch.manual_seed(0)
    y = torch.randn(frames, 1, 1, spokes, n, 1, dtype=torch.complex64)

    _finufft.reset_counters()
    x = bt.nufft(traj, y, adjoint=True, image_dims=(n, n, 1), B=basis)
    _all_finufft()

    phase = _phase_per_frame(traj, n, +1)
    data = y.numpy().reshape(frames, spokes, n)
    b = basis.numpy().reshape(coeffs, frames)
    ref = np.stack(
        [
            (np.conj(b[c])[:, None, None, None, None] * data[..., None, None] * phase).sum(
                axis=(0, 1, 2)
            )
            / n
            for c in range(coeffs)
        ]
    )
    got = x.numpy().reshape(ref.shape)
    _within_tolerance(got, ref)


@requires_finufft
def test_a_subspace_forward_over_a_per_frame_trajectory_matches_an_explicit_sum(in_tools):
    n, spokes, frames, coeffs = 16, 5, 4, 2
    traj, basis = _subspace(n, spokes, frames, coeffs)
    torch.manual_seed(0)
    img = torch.randn(coeffs, 1, 1, 1, 1, n, n, dtype=torch.complex64)

    _finufft.reset_counters()
    y = bt.nufft(traj, img, B=basis)
    _all_finufft()

    phase = _phase_per_frame(traj, n, -1)
    im = img.numpy().reshape(coeffs, n, n)
    b = basis.numpy().reshape(coeffs, frames)
    per_coeff = np.stack(
        [(phase * im[c][None, None, None]).sum(axis=(-1, -2)) / n for c in range(coeffs)]
    )
    ref = np.einsum("kt,ktsr->tsr", b, per_coeff)
    got = y.numpy().reshape(ref.shape)
    _within_tolerance(got, ref)


@requires_finufft
@pytest.mark.skipif(
    not (bartorch.cuda.available() and _finufft.cuda_available()),
    reason="this needs a CUDA device and the cufinufft package",
)
def test_a_subspace_adjoint_on_a_card_agrees_with_the_host(in_tools):
    n, spokes, frames, coeffs = 16, 5, 4, 2
    traj, basis = _subspace(n, spokes, frames, coeffs)
    torch.manual_seed(0)
    y = torch.randn(frames, 1, 1, spokes, n, 1, dtype=torch.complex64)

    _finufft.reset_counters()
    on_card = bt.nufft(traj.cuda(), y.cuda(), adjoint=True, image_dims=(n, n, 1), B=basis.cuda())
    _all_finufft()
    on_host = bt.nufft(traj, y, adjoint=True, image_dims=(n, n, 1), B=basis)
    # The two libraries agree to about 2e-3 on a grid this coarse at the
    # upsampling the substitution defaults to; on a 128 grid it is 1e-5.
    _sides_agree(on_card.cpu(), on_host)


@requires_finufft
def test_a_transform_finufft_cannot_serve_is_an_error_rather_than_barts_gridder():
    """Asking for FINUFFT and quietly getting BART would be the worst outcome.

    Images that vary across frames along the same axis the trajectory varies
    on need one transform per frame, which one plan cannot express.  That is a
    refusal by default, and BART's own operator answers it only when the
    caller says so.
    """
    n, spokes, frames, coils = 16, 5, 4, 2
    traj = bt.traj(x=n, y=spokes * frames, r=True).reshape(frames, spokes, n, 3)[:, None, None]
    torch.manual_seed(0)
    img = torch.randn(frames, 1, coils, 1, n, n, dtype=torch.complex64)

    assert not _finufft.fallback_allowed()
    with pytest.raises(bartorch.BartError, match="vary across frames"):
        bt.nufft(traj, img)

    # BART's own gridder still computes it, for whoever holds the two together.
    with _finufft.barts_own_gridder():
        _finufft.reset_counters()
        bt.nufft(traj, img)
        assert _finufft.operators_built() == (0, 1)


def test_enabling_without_finufft_says_so_rather_than_carrying_on(monkeypatch):
    monkeypatch.setattr(_finufft, "available", lambda: False)
    with pytest.raises(ImportError, match="finufft"):
        _finufft.use_in_tools(True)


@requires_finufft
def test_enabling_on_a_machine_with_a_device_needs_cufinufft(monkeypatch):
    """A card BART would use and no cuFINUFFT is a gap the caller should hear about."""
    from bartorch import _cuda

    monkeypatch.setattr(_cuda, "available", lambda: True)
    monkeypatch.setattr(_finufft, "cuda_available", lambda: False)
    with pytest.raises(ImportError, match="cufinufft"):
        _finufft.use_in_tools(True)


@requires_finufft
def test_barts_oversampling_is_finuffts_upsampling(in_tools):
    """``-o`` and ``upsampfac`` are the same number, so it is carried across.

    BART's own default is two, which leaves the choice to whatever ``enable``
    was told; anything else was asked for on purpose.
    """
    n, spokes = 64, 32
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ref = _dft(traj, img, n)

    for oversampling in (1.25, 1.5, 2.0):
        _finufft.reset_counters()
        A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False, oversampling=oversampling)
        _all_finufft()
        got = A(img).numpy().reshape(ref.shape)
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel < 1e-3, (oversampling, rel)


@requires_finufft
def test_a_kernel_width_asked_for_buys_the_accuracy_that_width_buys(in_tools):
    """``-w`` is a count of grid points, and so is FINUFFT's ns.

    FINUFFT has no field to be told a width: it sizes ns from the tolerance by
    ``ns = ceil(ln(tolfac/tol) / (pi sqrt(1 - 1/sigma)) + 1)``, so the width
    asked for is carried across by inverting that.  A narrower kernel has to
    come out less accurate than a wider one, which is the whole content of the
    flag.
    """
    n, spokes = 64, 48
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ref = _dft(traj, img, n)
    nref = np.linalg.norm(ref)

    # Past about seven grid points the kernel is no longer what limits a
    # single-precision transform, so the widths that say anything are narrow.
    errors = {}
    for width in (2.0, 3.0, 4.0):
        _finufft.reset_counters()
        A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False, width=width)
        _all_finufft()
        got = A(img).numpy().reshape(ref.shape)
        errors[width] = np.linalg.norm(got - ref) / nref

    assert errors[2.0] > errors[3.0] > errors[4.0], errors
    # A width of four at a quarter over is about two thousandths.  It used to
    # read better than that because the tolerance a width was asked for landed
    # exactly on the boundary between two widths, and rounding handed back a
    # kernel one wider than the one requested.
    assert errors[4.0] < 5e-3, errors


@requires_finufft
def test_precision_can_be_traded_for_a_transform_that_fits():
    """The default is the cheap transform, and precision is what is asked for.

    A reconstruction is not made better by a transform an order more accurate
    than the data going into it, and a three-dimensional subspace problem on a
    laptop is only feasible at a tolerance and a grid somebody chose.  So the
    default is a thousandth on a grid a quarter over, and a caller who wants
    the textbook grid and six digits asks for them.
    """
    n, spokes = 64, 48
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ref = _dft(traj, img, n)
    nref = np.linalg.norm(ref)

    def error():
        _finufft.reset_counters()
        A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
        _all_finufft()
        return np.linalg.norm(A(img).numpy().reshape(ref.shape) - ref) / nref

    try:
        _finufft.use_in_tools(True)
        cheap = error()
        assert _finufft.tolerance() == pytest.approx(1e-3), "a thousandth by default"
        assert _finufft.upsampling() == pytest.approx(1.25), "a quarter over by default"

        _finufft.use_in_tools(True, tolerance=1e-6, upsampling=2.0)
        careful = error()
        assert _finufft.tolerance() == pytest.approx(1e-6)
        assert _finufft.upsampling() == pytest.approx(2.0)

        # What was asked for is what came back: each loose by about the amount
        # asked for rather than by an unbounded amount.
        assert careful < cheap < 1e-2, (careful, cheap)
    finally:
        _finufft.use_in_tools(True)


@requires_finufft
def test_nothing_reaches_barts_gridder_without_having_been_sent_there():
    """The substitution installs itself, and a decline is an error either way.

    A caller who never mentions FINUFFT still gets it, because the alternative
    is an answer an order further from the transform and several times slower
    with nothing to say so.  BART's own gridder is one call away and no closer.
    """
    n, spokes = 32, 16
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)

    _finufft.reset_counters()
    bt.nufft(traj, img)
    _all_finufft()
    assert not _finufft.fallback_allowed()

    # Something FINUFFT cannot serve, with nothing having been asked for.
    frames = 4
    varying = bt.traj(x=n, y=5 * frames, r=True).reshape(frames, 5, n, 3)[:, None, None]
    per_frame = torch.zeros(frames, 1, 2, 1, n, n, dtype=torch.complex64)
    with pytest.raises(bartorch.BartError, match="FINUFFT cannot serve"):
        bt.nufft(varying, per_frame)


@requires_finufft
def test_a_subspace_operator_needs_no_tool_and_no_fallback():
    """The operator layer can say what the tools can, or it is a way back to BART.

    A caller who wants a subspace transform without BART's command mains had
    no way to ask for one, and shapes that implied it were refused, which left
    the tool path as the only route.  The basis belongs to the operator
    because the normal is a point spread function over it.
    """
    n, spokes, frames, coeffs = 16, 5, 4, 2
    traj, basis = _subspace(n, spokes, frames, coeffs)
    torch.manual_seed(0)
    img = torch.randn(coeffs, 1, 1, 1, 1, n, n, dtype=torch.complex64)

    _finufft.reset_counters()
    A = LinearOperator.nufft(
        traj,
        (coeffs, 1, 1, 1, 1, n, n),
        kspace_shape=(frames, 1, 1, spokes, n, 1),
        basis=basis,
        toeplitz=False,
    )
    _all_finufft()
    assert not _finufft.fallback_allowed()

    phase = _phase_per_frame(traj, n, -1)
    im = img.numpy().reshape(coeffs, n, n)
    b = basis.numpy().reshape(coeffs, frames)
    per_coeff = np.stack(
        [(phase * im[c][None, None, None]).sum(axis=(-1, -2)) / n for c in range(coeffs)]
    )
    ref = np.einsum("kt,ktsr->tsr", b, per_coeff)
    got = A(img).numpy().reshape(ref.shape)
    _within_tolerance(got, ref)


@requires_finufft
def test_oversampling_and_width_compose(in_tools):
    """``-o`` and ``-w`` are both carried across, and mean what they mean together.

    ``-o`` is FINUFFT's upsampfac outright.  ``-w`` has no field of its own, so
    it becomes the tolerance that yields that kernel width -- at the upsampling
    in force, which is why the same width buys less on a smaller grid: three
    points of kernel on a grid a quarter over really is coarser than three on
    one twice over.
    """
    n, spokes = 64, 48
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ref = _dft(traj, img, n)
    nref = np.linalg.norm(ref)

    def error(**kw):
        A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False, **kw)
        _finufft.reset_counters()
        out = A(img)
        return np.linalg.norm(out.numpy().reshape(ref.shape) - ref) / nref

    # Half over, not twice: two is BART's own default for `-o` and
    # `nufft_conf_s` carries no way to tell it from a caller who never set it,
    # and by two and a half a wider grid already gives what the kernel would,
    # so the width stops changing anything that can be measured.
    wide_on_a_half = error(oversampling=1.5, width=4.0)
    narrow_on_a_half = error(oversampling=1.5, width=3.0)
    wide_on_a_quarter = error(oversampling=1.25, width=4.0)

    assert narrow_on_a_half > wide_on_a_half, "a narrower kernel is a looser transform"
    assert wide_on_a_quarter > wide_on_a_half, "the same width on a smaller grid is coarser"
    assert wide_on_a_half < 2e-3


@requires_finufft
def test_the_grid_a_caller_asks_for_is_the_grid_they_get(in_tools):
    """Two is a factor like any other, not a way of saying nothing.

    ``nufft_conf_s`` carries two as BART's own default, so a conf that says
    two says nothing about whether anybody asked for it.  Zero is what says
    nobody did, and BART gets its two back before it sees the conf.
    """
    n, spokes = 64, 48
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ref = _dft(traj, img, n)
    nref = np.linalg.norm(ref)

    def error(**kw):
        A = LinearOperator.nufft(traj, (1, n, n), toeplitz=False, **kw)
        return np.linalg.norm(A(img).numpy().reshape(ref.shape) - ref) / nref

    assert _finufft.upsampling() == pytest.approx(1.25), "a quarter over by default"
    default = error()
    assert error(oversampling=1.25) == pytest.approx(default), "the default, said out loud"

    # The textbook grid is finer than the default, and asking for it works.
    assert error(oversampling=2.0) < default, "asking for two has to give two"


@requires_finufft
def test_a_tools_oversampling_does_not_outlive_the_command(in_tools):
    """``nufft_conf_options`` is a global, and this process runs more than one
    command through it."""
    n, spokes = 64, 48
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    ref = _dft(traj, img, n)

    def error(**kw):
        y = bt.nufft(traj, img, **kw)
        return np.linalg.norm(y.numpy().reshape(ref.shape) - ref) / np.linalg.norm(ref)

    before = error()
    assert error(o=2.0) < before, "`-o 2` is two"
    assert error() == pytest.approx(before), "and is gone by the next command"


@requires_finufft
def test_a_tool_that_calibrates_gets_the_careful_transform(in_tools):
    """Coil sensitivities are what everything after them is built on.

    ``nlinv`` fits them at a resolution where the textbook grid costs a few
    megabytes, so it gets FINUFFT's own tolerance on it rather than the cheap
    default the rest of the library runs at.
    """
    from bartorch.core import graph

    assert "nlinv" in graph._CALIBRATES

    n, spokes, coils = 32, 32, 2
    traj = bt.traj(x=n, y=spokes, r=True)
    ksp = bt.nufft(traj, bt.phantom([n, n], ncoils=coils))

    careful = bt.nlinv(ksp, t=traj, iter_=4)

    was = graph._CALIBRATES
    graph._CALIBRATES = frozenset()
    try:
        cheap = bt.nlinv(ksp, t=traj, iter_=4)
    finally:
        graph._CALIBRATES = was

    assert float((careful - cheap).abs().max()) > 0.0, "the careful transform changed nothing"
    # And the library is left as it was found.
    assert _finufft.tolerance() == pytest.approx(1e-3)
    assert _finufft.upsampling() == pytest.approx(1.25)


@requires_finufft
def test_a_point_spread_function_is_the_substituted_transforms(in_tools):
    """A PSF is an adjoint transform of ones, and that transform is FINUFFT's.

    BART reaches it through `nufft.c`'s own `nufft_create2`, which the rename
    sends to its gridder along with everything else in that file, so
    `csrc/psf.c` answers to the three entry points instead.  What that buys is
    in the numbers: on the un-doubled grid BART's own PSF is two per cent from
    the sum it is supposed to be, and this one is at its tolerance.
    """
    from bartorch.tools import _generated as g

    n, spokes = 16, 12
    traj = bt.traj(x=n, y=spokes, r=True)

    k = traj.numpy().real.reshape(-1, 3)
    x = np.arange(n) - n // 2
    phase = np.exp(
        2j
        * np.pi
        * (
            k[:, 0][:, None, None] * x[None, None, :] / n
            + k[:, 1][:, None, None] * x[None, :, None] / n
        )
    )
    ref = phase.sum(axis=0)

    # After configuring, not before: putting the substitution in place checks
    # itself against BART's gridder, and builds one of each doing it.
    _finufft.use_in_tools(True, tolerance=1e-6, upsampling=2.0)
    _finufft.reset_counters()
    try:
        ours = g.psf(traj).numpy()
        _all_finufft()
    finally:
        _finufft.use_in_tools(True)

    # Each carries its own scaling; what is compared is the function.
    scale = np.vdot(ours, ref) / np.vdot(ours, ours)
    assert np.linalg.norm(ours * scale - ref) / np.linalg.norm(ref) < 1e-5


@requires_finufft
@pytest.mark.parametrize("flags", [{}, {"oversampled": True}, {"oversampled_decomposed": True}])
def test_every_psf_the_tool_offers_is_served(in_tools, flags):
    """`compute_psf`, `compute_psf2` and `compute_psf2_decomposed`, one each.

    The decomposed one takes a set of frequencies at a time, each with its own
    shifted trajectory and its own image, which is a stack of transforms
    rather than one plan over frames.
    """
    from bartorch.tools import _generated as g

    n, spokes = 16, 12
    traj = bt.traj(x=n, y=spokes, r=True)

    _finufft.reset_counters()
    ours = g.psf(traj, **flags)
    built, bart = _finufft.operators_built()
    assert bart == 0, _finufft.decline_reason()
    assert built >= 1

    with _finufft.barts_own_gridder():
        theirs = g.psf(traj, **flags)

    assert ours.shape == theirs.shape


@requires_finufft
def test_the_toeplitz_normal_is_the_transform_pair_it_stands_for(in_tools):
    """A^H A as one convolution, against A^H A as two transforms.

    The point spread function it convolves with is computed here now, so what
    holds the two together is only the tolerance the transforms were planned
    with -- and the gap closes with it, which a function that was subtly the
    wrong one would not do.
    """
    torch.manual_seed(0)
    n, spokes = 64, 96
    traj = bt.traj(x=n, y=spokes, r=True)
    x = torch.randn(1, n, n, dtype=torch.complex64)

    errors = {}
    try:
        for eps, upsampling in ((1e-3, 1.25), (1e-6, 2.0)):
            _finufft.use_in_tools(True, tolerance=eps, upsampling=upsampling)
            A = LinearOperator.nufft(traj, (1, n, n), toeplitz=True)
            errors[eps] = float((A.normal(x) - A.adjoint(A(x))).norm() / A.adjoint(A(x)).norm())
    finally:
        _finufft.use_in_tools(True)

    assert errors[1e-3] < 5e-3
    assert errors[1e-6] < 1e-5
    assert errors[1e-6] < errors[1e-3], errors


@requires_finufft
def test_nothing_builds_barts_own_nufft(in_tools):
    """The counter of BART's own operators is the whole claim.

    Every route to one is counted: a decline that falls back, and a normal
    whose point spread function BART would have to grid for itself.
    """
    n, spokes, coils = 32, 32, 2
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n], ncoils=coils)
    ksp = bt.nufft(traj, img)
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    for name, run in (
        ("nufft", lambda: bt.nufft(traj, img)),
        ("nufft -i", lambda: bt.nufft(traj, ksp, inverse=True, image_dims=(n, n, 1))),
        ("pics", lambda: bt.pics(ksp, maps, t=traj)),
        ("nlinv", lambda: bt.nlinv(ksp, t=traj, iter_=3)),
        ("operator", lambda: LinearOperator.nufft(traj, (1, n, n), toeplitz=True)),
    ):
        _finufft.reset_counters()
        run()
        assert _finufft.operators_built()[1] == 0, (name, _finufft.decline_reason())


@requires_finufft
@pytest.mark.parametrize(
    "mode",
    [None, "lowmem", "no-precomp", "decomposed-psf", "real-psf", "compress-psf", "zero-mem"],
)
def test_every_way_bart_stores_a_point_spread_function_is_served(in_tools, mode):
    """Nothing was taken away, and none of it costs a gridding.

    `conf.nopsf` is what keeps BART from computing a function of its own --
    the switch `pics --psf_import` uses to bring one in from outside -- so
    what it does with the function afterwards is all still BART's: floats for
    a real one, the entries that are not zero for a compressed one, the
    oversampled grid and the linear phases around both.
    """
    n, spokes, coils = 32, 48, 2
    traj = bt.traj(x=n, y=spokes, r=True)
    img = bt.phantom([n, n], ncoils=coils)
    ksp = bt.nufft(traj, img)
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    reference = bt.pics(ksp, maps, t=traj)

    _finufft.reset_counters()
    kwargs = {} if mode is None else {"nufft_conf": mode}
    out = bt.pics(ksp, maps, t=traj, **kwargs)

    assert _finufft.operators_built()[1] == 0, _finufft.decline_reason()

    # `zero-mem` is a parenthesised flag in BART's own help, and its Toeplitz
    # normal does not reconstruct there either: BART's own is nearly two from
    # BART's own default.  What is claimed for it here is the interception.
    if mode == "zero-mem":
        return

    # Compression throws away what its mask does not cover.  The mask is
    # spread with FINUFFT's kernel, so it covers where this function has
    # signal rather than where BART's would have had it.
    # The decomposed one computes the function a set of frequencies at a
    # time, so its error is the tolerance compounded over the sets rather
    # than paid once.
    bound = 5e-2 if mode in ("compress-psf", "decomposed-psf") else 1e-2
    assert float((out - reference).abs().max() / reference.abs().max()) < bound


@requires_finufft
def test_an_upper_triangular_subspace_function_is_served(in_tools):
    """Half of a Hermitian function is still one this computes.

    `compute_psf2` takes the flag, so the only difference here is the shape it
    comes back in -- and `nufft_create_normal` asserts that shape against the
    linear phases it would have built, which is what checks it.
    """
    from bartorch.tools import _generated as g

    n, spokes, frames, coeffs, coils = 16, 5, 4, 2, 2
    traj, basis = _subspace(n, spokes, frames, coeffs)
    torch.manual_seed(0)
    k = torch.randn(frames, 1, coils, spokes, n, 1, dtype=torch.complex64)
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    whole = g.pics(k, maps, t=traj, B=basis, i=5)

    _finufft.reset_counters()
    half = g.pics(k, maps, t=traj, B=basis, i=5, nufft_conf="upper-triag-psf")

    assert _finufft.operators_built()[1] == 0, _finufft.decline_reason()
    torch.testing.assert_close(half, whole, rtol=1e-4, atol=1e-4)


@requires_finufft
def test_a_compressed_function_keeps_what_this_transform_put_there(in_tools):
    """The mask is the footprint of the kernel that spread the function.

    BART finds it by spreading the sampling pattern with its own Kaiser-Bessel
    kernel, which is the wrong footprint once the function is FINUFFT's.
    Spreading the pattern with FINUFFT's kernel instead -- `spreadinterponly`,
    one set of frequencies at a time so the doubled grid is never allocated --
    covers where the function actually has signal, and the reconstruction says
    so.
    """
    n, spokes, coils = 32, 48, 2
    traj = bt.traj(x=n, y=spokes, r=True)
    ksp = bt.nufft(traj, bt.phantom([n, n], ncoils=coils))
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    def cost():
        plain = bt.pics(ksp, maps, t=traj)
        compressed = bt.pics(ksp, maps, t=traj, nufft_conf="compress-psf")
        return float((compressed - plain).abs().max() / plain.abs().max())

    ours = cost()
    with _finufft.barts_own_gridder():
        theirs = cost()

    assert ours < theirs, (ours, theirs)


@requires_finufft
def test_the_mask_lands_where_barts_does(in_tools, caplog):
    """Given the same kernel, the two masks keep the same points.

    Accuracy alone would not catch a mask that is displaced rather than
    mis-sized: one that keeps the wrong points but more of them can still
    reconstruct well.  The compression rate is what catches it, and it only
    means something when both are spread with the same footprint -- BART's
    width 6 at an oversampling of 2 covers 3 cells of the image grid, and so
    does FINUFFT's kernel at an upsampling of 2 and a tolerance that buys
    ns = 6.
    """
    n, spokes, coils = 64, 64, 2
    traj = bt.traj(x=n, y=spokes, r=True)
    ksp = bt.nufft(traj, bt.phantom([n, n], ncoils=coils))
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    def kept():
        return _percent_kept(caplog, ksp, maps, traj)

    try:
        _finufft.use_in_tools(True, tolerance=4.5e-6, upsampling=2.0)
        ours = kept()
        with _finufft.barts_own_gridder():
            theirs = kept()
    finally:
        _finufft.use_in_tools(True)

    # Rounding a width to whole cells leaves this one a little wider; a mask
    # in the wrong place would not be within a few points of BART's.
    assert abs(ours - theirs) <= 5, (ours, theirs)


def _percent_kept(caplog, ksp, maps, traj):
    """What fraction of the grid a compressed function keeps, off the log."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="bartorch.bart"):
        bartorch.set_debug_level(4)
        try:
            bt.pics(ksp, maps, t=traj, nufft_conf="compress-psf")
        finally:
            bartorch.set_debug_level(1)
    percent = [
        int(m.split("to")[1].strip().rstrip("%")) for m in caplog.messages if "Compressing PSF" in m
    ]
    assert percent, caplog.messages
    return percent[0]


@requires_finufft
def test_the_mask_is_the_width_and_not_the_upsampling(in_tools, caplog):
    """A mask is set by its width and the geometry, and by nothing else.

    Which is the thing that breaks if the function and the mask disagree about
    where a sample lands: the upsampling sizes FINUFFT's own fine grid and has
    nothing to say about the image grid the mask lives on, so two tolerances
    that buy the same width off different upsamplings have to keep the same
    points.  A tolerance of a millionth at an upsampling of two and one of a
    thousandth at a quarter over both buy a width of four.
    """
    n, spokes, coils = 64, 64, 2
    traj = bt.traj(x=n, y=spokes, r=True)
    ksp = bt.nufft(traj, bt.phantom([n, n], ncoils=coils))
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    try:
        _finufft.use_in_tools(True, tolerance=1e-6, upsampling=2.0)
        wide = _percent_kept(caplog, ksp, maps, traj)

        _finufft.use_in_tools(True, tolerance=1e-3, upsampling=1.25)
        cheap = _percent_kept(caplog, ksp, maps, traj)

        # And narrower is narrower: a width of three keeps fewer.
        _finufft.use_in_tools(True, tolerance=4.5e-6, upsampling=2.0)
        narrow = _percent_kept(caplog, ksp, maps, traj)
    finally:
        _finufft.use_in_tools(True)

    assert wide == cheap, (wide, cheap)
    assert narrow < wide, (narrow, wide)


@requires_finufft
def test_the_toeplitz_kernel_is_the_doubled_grid_whatever_the_upsampling(in_tools):
    """The embedding's grid is twice the image, and the kernel's is not.

    Two things are called an oversampling here and only one of them is the
    Toeplitz grid: the function is over 2N whatever FINUFFT spreads on
    underneath, held as 2^d copies of N because that is how `nufft.c` stores
    it.  The upsampling buys accuracy in the transform that builds it and
    nothing else, so a looser one has to give the same function to within the
    tolerance rather than a different-shaped one.
    """
    from bartorch.tools import _generated as g

    n = 32
    traj = bt.traj(x=n, y=48, r=True)

    kernels = {}
    try:
        for label, eps, upsampling in (("cheap", 1e-3, 1.25), ("careful", 1e-6, 2.0)):
            _finufft.use_in_tools(True, tolerance=eps, upsampling=upsampling)
            kernels[label] = g.psf(traj, oversampled=True)
    finally:
        _finufft.use_in_tools(True)

    for label, kernel in kernels.items():
        assert kernel.numel() == (2 * n) ** 2, (label, tuple(kernel.shape))
        assert kernel.shape[0] == 4, (label, tuple(kernel.shape))  # 2^d sets
        assert kernel.shape[-2:] == (n, n), (label, tuple(kernel.shape))

    assert kernels["cheap"].shape == kernels["careful"].shape

    cheap, careful = kernels["cheap"].numpy(), kernels["careful"].numpy()
    assert np.linalg.norm(cheap - careful) / np.linalg.norm(careful) < 5 * 1e-3


@requires_finufft
def test_a_plan_lives_exactly_as_long_as_what_asked_for_it(in_tools):
    """Every plan is freed by the thing that made it, and nothing outlives it.

    A plan holds its own workspace and, on a card, device memory, so one left
    behind by an operator, by a point spread function or by the spreading a
    compressed one is masked with would accumulate over a solve.  The count is
    what says so: reading the process instead would say nothing, because
    FINUFFT's own multithreaded execute retains about a kilobyte per thread on
    every call and that swamps anything a plan costs.
    """
    n, spokes, coils = 32, 48, 2
    traj = bt.traj(x=n, y=spokes, r=True)
    image = bt.phantom([n, n], ncoils=coils)
    ksp = bt.nufft(traj, image)
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5
    x = torch.randn(1, n, n, dtype=torch.complex64)

    assert _finufft.live_plans() == 0

    held = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    held(x)
    assert _finufft.live_plans() > 0, "an operator that has transformed holds a plan"
    del held
    assert _finufft.live_plans() == 0, "and gives it back when it is freed"

    # A Toeplitz operator makes more of them -- the pair, the transform behind
    # the point spread function, and one spreader per frequency set for the
    # mask -- and each is freed where it was made.
    LinearOperator.nufft(traj, (1, n, n), toeplitz=True).normal(x)
    assert _finufft.live_plans() == 0

    for run in (
        lambda: bt.nufft(traj, image),
        lambda: bt.nufft(traj, ksp, adjoint=True),
        lambda: bt.psf(traj),
        lambda: bt.pics(ksp, maps, t=traj),
        lambda: bt.pics(ksp, maps, t=traj, no_toeplitz=True),
        lambda: bt.nlinv(ksp, t=traj, iter_=3),
    ):
        run()
        assert _finufft.live_plans() == 0


@requires_finufft
@pytest.mark.skipif(
    not bartorch.cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)
def test_a_device_plan_is_given_back_too(in_tools):
    """A cuFINUFFT plan holds device memory, which is the scarcer of the two."""
    n, spokes, coils = 32, 48, 2
    traj = bt.traj(x=n, y=spokes, r=True).cuda()
    ksp = bt.nufft(traj, bt.phantom([n, n], ncoils=coils).cuda())
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()
    x = torch.randn(1, n, n, dtype=torch.complex64, device="cuda")

    assert _finufft.live_plans() == 0

    LinearOperator.nufft(traj, (1, n, n), toeplitz=True).normal(x)
    assert _finufft.live_plans() == 0

    bt.pics(ksp, maps, t=traj)
    assert _finufft.live_plans() == 0


@requires_finufft
def test_one_thread_count_covers_bart_and_the_transform(in_tools):
    """``set_num_threads`` means the process, not just BART.

    FINUFFT takes a thread per physical core unless it is told otherwise, so
    a caller who has limited BART to leave room for something else would
    otherwise still find the transform taking the whole machine.  Zero is the
    state it starts in and the way back to it, which is why it has a setter of
    its own: BART has no count that means "choose for me".
    """
    assert _finufft.threads() == 0, "FINUFFT chooses for itself until it is told"

    try:
        bartorch.set_num_threads(2)
        assert _finufft.threads() == 2

        _finufft.set_threads(0)
        assert _finufft.threads() == 0, "and can be put back without BART losing its count"

        # A transform still runs whichever way round it is set.
        n = 32
        traj = bt.traj(x=n, y=48, r=True)
        image = bt.phantom([n, n]).reshape(1, n, n)
        reference = bt.nufft(traj, image)

        _finufft.set_threads(1)
        one = bt.nufft(traj, image)
    finally:
        _finufft.set_threads(0)
        bartorch.set_num_threads(os.cpu_count() or 1)

    torch.testing.assert_close(one, reference, rtol=1e-4, atol=1e-5)
