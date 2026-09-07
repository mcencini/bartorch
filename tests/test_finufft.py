"""The FINUFFT-backed NUFFT, as an operator and underneath BART's own tools,
against an explicit discrete Fourier sum and against BART's own gridder.
"""

import numpy as np
import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _finufft
from bartorch.ops import LinearOperator

requires_finufft = pytest.mark.skipif(not _finufft.available(), reason="finufft is not installed")


def _inner(a, b):
    return torch.vdot(a.flatten(), b.flatten()).real.item()


@requires_finufft
def test_matches_an_explicit_dft_on_a_radial_trajectory():
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    y = LinearOperator.finufft(traj, (1, n, n))(img)

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
    n = 64
    traj = bt.traj(x=n, y=32, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    bart = LinearOperator.nufft(traj, (1, n, n), toeplitz=False)
    fast = LinearOperator.finufft(traj, (1, n, n))
    assert fast.ishape == bart.ishape and fast.oshape == bart.oshape
    a, b = bart(img), fast(img)
    assert (a - b).norm().item() / a.norm().item() < 5e-3


@requires_finufft
def test_adjoint_identity_holds():
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    A = LinearOperator.finufft(traj, (1, n, n))
    x = torch.randn(*A.ishape, dtype=torch.complex64)
    y = torch.randn(*A.oshape, dtype=torch.complex64)
    assert _inner(A(x), y) == pytest.approx(_inner(x, A.adjoint(y)), rel=1e-3)


@requires_finufft
def test_it_carries_coils_through_one_plan():
    n, ncoils = 32, 4
    traj = bt.traj(x=n, y=16, r=True)
    A = LinearOperator.finufft(traj, (ncoils, n, n))
    assert A.oshape[0] == ncoils
    x = torch.randn(ncoils, n, n, dtype=torch.complex64)
    y = A(x)
    # Each coil must transform independently of the others.
    single = LinearOperator.finufft(traj, (1, n, n))
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
    A = LinearOperator.finufft(traj, (1, n, n))
    y = A(img)
    x = A.lstsq(y, lambda_=1e-3, maxiter=30)
    assert x.shape == img.shape
    assert (x - img).norm().item() / img.norm().item() < 0.5


@pytest.fixture
def in_tools():
    """BART's tools computing their NUFFT with FINUFFT, for one test."""
    if not _finufft.use_in_tools():
        pytest.skip("the substitution declined to install itself")
    yield
    _finufft.use_in_tools(False)


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

    assert _finufft.operators_built() == (1, 0), _finufft.decline_reason()
    ref = _dft(traj, img, n)
    assert np.linalg.norm(y.numpy().reshape(ref.shape) - ref) / np.linalg.norm(ref) < 1e-4


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

    assert _finufft.operators_built() == (1, 0), _finufft.decline_reason()
    ref = _dft(traj, img, n) * weights.numpy().reshape(spokes, n)
    assert np.linalg.norm(y.numpy().reshape(ref.shape) - ref) / np.linalg.norm(ref) < 1e-4

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
    assert _finufft.operators_built() == (1, 0), _finufft.decline_reason()

    _finufft.use_in_tools(False)
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
def test_turning_it_off_gives_the_tools_barts_operator_back():
    _finufft.use_in_tools(False)
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)

    _finufft.reset_counters()
    bt.nufft(traj, img)

    assert not _finufft.used_in_tools()
    assert _finufft.operators_built() == (0, 1)
    assert _finufft.decline_reason() == "FINUFFT is not in use"


@requires_finufft
def test_the_operator_is_unaffected_by_the_substitution():
    n = 32
    traj = bt.traj(x=n, y=16, r=True)
    img = bt.phantom([n, n]).reshape(1, n, n)
    A = LinearOperator.finufft(traj, (1, n, n))

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
    assert _finufft.operators_built() == (1, 0), _finufft.decline_reason()
    ref = _dft(traj.cpu(), image.cpu(), n)
    assert np.linalg.norm(y.cpu().numpy().reshape(ref.shape) - ref) / np.linalg.norm(ref) < 1e-4


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
    assert _finufft.operators_built() == (1, 0), _finufft.decline_reason()

    on_card = A(image.cuda())
    on_host = A(image)
    assert on_card.device.type == "cuda" and on_host.device.type == "cpu"

    ref = _dft(traj.cpu(), image, n)
    got = on_card.cpu().numpy().reshape(ref.shape)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4
    torch.testing.assert_close(on_card.cpu(), on_host, rtol=1e-4, atol=1e-4)


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
    assert _finufft.operators_built() == (1, 0), _finufft.decline_reason()

    y = A(image)
    assert y.shape[0] == frames

    # Every frame holds the same source, so every frame holds the same samples.
    torch.testing.assert_close(y[0], y[-1])
    ref = _dft(traj, image[:1], n)
    got = y[0].numpy().reshape(ref.shape)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4
