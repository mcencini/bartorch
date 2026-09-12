"""The SENSE operators, which walk their coils a slab at a time.

The coils are independent until the sum that ends the adjoint, so the operator
applies a slab of sensitivities, asks the transform for that slab, and sums it
into the answer.  What that changes is residency, not arithmetic, so the test
is that it changes nothing: the same reconstruction whatever the slab, held
against BART's own operator over every coil at once.
"""

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _dispatch, linop, optim
from bartorch._lib import library


@pytest.fixture
def restore_batch():
    was = _dispatch.coil_batch()
    yield
    _dispatch.set_coil_batch(was)


def _problem(n=32, coils=8, seed=0):
    torch.manual_seed(seed)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()
    image = (bt.phantom([n, n])[None] * maps).reshape(coils, 1, n, n)
    return image, maps.reshape(1, coils, 1, n, n)


# BART's own operator, over every coil at once, is what each slab has to agree
# with -- so the reference is the arrangement being replaced, not another slab.
BATCHES = [1, 2, 4, 8, 16]


@pytest.mark.parametrize("batch", BATCHES)
def test_a_cartesian_reconstruction_is_the_same_whatever_the_slab(batch, restore_batch):
    image, maps = _problem()
    kspace = bartorch.fft(image, axes=(-2, -1))

    _dispatch.set_coil_batch(0)
    reference = bt.pics(kspace, maps, maxiter=30)

    _dispatch.set_coil_batch(batch)
    torch.testing.assert_close(bt.pics(kspace, maps, maxiter=30), reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch", BATCHES)
def test_a_non_cartesian_reconstruction_is_the_same_whatever_the_slab(batch, restore_batch):
    image, maps = _problem()
    traj = bt.traj(x=32, y=48, r=True)
    kspace = bartorch.nufft(image, traj)

    _dispatch.set_coil_batch(0)
    reference = bt.pics(kspace, maps, t=traj, maxiter=30)

    _dispatch.set_coil_batch(batch)
    got = bt.pics(kspace, maps, t=traj, maxiter=30)

    scale = float(reference.abs().max())
    assert float((got - reference).abs().max()) / scale < 1e-4


def test_the_slab_is_actually_taken(restore_batch):
    """A slab that silently fell back to BART's chain would pass every test above."""
    lib = library()
    image, maps = _problem()
    traj = bt.traj(x=32, y=48, r=True)
    kspace = bartorch.nufft(image, traj)

    _dispatch.set_coil_batch(0)
    lib.bartorch_sense_reset_counters()
    bt.pics(kspace, maps, t=traj, maxiter=3)
    assert (lib.bartorch_sense_counter(0), lib.bartorch_sense_counter(1)) == (0, 1)

    _dispatch.set_coil_batch(2)
    lib.bartorch_sense_reset_counters()
    bt.pics(kspace, maps, t=traj, maxiter=3)
    bt.pics(bartorch.fft(image, axes=(-2, -1)), maps, maxiter=3)
    assert lib.bartorch_sense_counter(0) == 2, "both SENSE operators walk their coils"
    assert lib.bartorch_sense_counter(1) == 0


def test_a_single_coil_goes_back_to_barts_own_operator(restore_batch):
    """There is nothing to slab, and an operator that pretended otherwise
    would pay for a loop of one."""
    lib = library()
    n = 32
    image = bt.phantom([n, n]).reshape(1, 1, n, n)
    maps = torch.ones(1, 1, 1, n, n, dtype=torch.complex64)

    _dispatch.set_coil_batch(1)
    lib.bartorch_sense_reset_counters()
    bt.pics(bartorch.fft(image, axes=(-2, -1)), maps, maxiter=3)
    assert lib.bartorch_sense_counter(0) == 0
    assert lib.bartorch_sense_counter(1) == 1


def test_the_setting_reports_itself(restore_batch):
    _dispatch.set_coil_batch(3)
    assert _dispatch.coil_batch() == 3
    _dispatch.set_coil_batch(0)
    assert _dispatch.coil_batch() == 0


def test_the_batch_is_the_operators_own(restore_batch):
    """Building an encoding leaves the default BART's tools use untouched."""
    lib = library()
    n, coils = 16, 4
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    _dispatch.set_coil_batch(1)

    lib.bartorch_sense_reset_counters()
    linop.CartesianSense(maps, (coils, n, n), coil_batch=0)
    assert (lib.bartorch_sense_counter(0), lib.bartorch_sense_counter(1)) == (0, 1)

    linop.CartesianSense(maps, (coils, n, n), coil_batch=2)
    assert lib.bartorch_sense_counter(0) == 1
    assert _dispatch.coil_batch() == 1


# --- sensitivities held as kernels ------------------------------------------


def _smooth_bank(n=32, coils=4, size=8, seed=0):
    """A bank that is band-limited by construction, and its kernels."""
    torch.manual_seed(seed)
    kernels = torch.randn(coils, size, size, dtype=torch.complex64)
    return kernels, bartorch.kernels_to_maps(kernels, (n, n))


def test_kernels_and_maps_are_the_same_thing_from_either_side():
    """Padding a cropped unitary spectrum back on to its own grid is a mask.

    So a bank that is band-limited to the kernel survives the round trip
    exactly, which is what says the two conventions -- where the centre is,
    and how the transform is normalised -- agree.
    """
    kernels, maps = _smooth_bank()
    again = bartorch.maps_to_kernels(maps, 8)
    torch.testing.assert_close(again, kernels, rtol=1e-5, atol=1e-5)


def test_an_odd_kernel_is_centred_where_the_transform_puts_it():
    """``md_resize_center`` centres at ``dim / 2``, so the offset between two
    sizes is the difference of their halves, not half their difference."""
    torch.manual_seed(0)
    kernels = torch.randn(4, 7, 7, dtype=torch.complex64)
    maps = bartorch.kernels_to_maps(kernels, (32, 32))
    torch.testing.assert_close(bartorch.maps_to_kernels(maps, 7), kernels, rtol=1e-5, atol=1e-5)


def test_a_kernel_bank_applies_as_the_maps_it_stands_for():
    """The operator inflates a slab at a time; that has to be the same
    operator as the one over the whole bank."""
    n, coils = 32, 4
    kernels, maps = _smooth_bank(n=n, coils=coils)
    x = bt.phantom([n, n]).reshape(1, n, n)

    dense = linop.CartesianSense(maps, (coils, n, n))
    compact = linop.CartesianSense(kernels, (coils, n, n), kernels=True)

    assert dense.ishape == compact.ishape
    assert dense.oshape == compact.oshape
    torch.testing.assert_close(compact(x), dense(x), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        compact.adjoint(dense(x)), dense.adjoint(dense(x)), rtol=1e-4, atol=1e-5
    )


def test_a_kernel_bank_applies_off_the_grid_too():
    n, coils = 32, 4
    kernels, maps = _smooth_bank(n=n, coils=coils)
    traj = bt.traj(x=n, y=48, r=True)
    x = bt.phantom([n, n]).reshape(1, n, n)

    dense = linop.NoncartesianSense(maps, (coils, n, n), traj=traj)
    compact = linop.NoncartesianSense(kernels, (coils, n, n), kernels=True, traj=traj)

    torch.testing.assert_close(compact(x), dense(x), rtol=1e-3, atol=1e-4)


def test_the_operator_is_the_sensitivities_and_the_transform():
    """Held against the tools, which are not this operator."""
    n, coils = 32, 4
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    x = bt.phantom([n, n]).reshape(1, n, n)
    coil_images = (x * maps).reshape(coils, 1, n, n)

    grid = linop.CartesianSense(maps, (coils, n, n))
    torch.testing.assert_close(
        grid(x).reshape(coils, 1, n, n),
        bartorch.fft(coil_images, axes=(-2, -1), unitary=True),
        rtol=1e-4,
        atol=1e-5,
    )

    traj = bt.traj(x=n, y=48, r=True)
    off = linop.NoncartesianSense(maps, (coils, n, n), traj=traj)
    torch.testing.assert_close(
        off(x).reshape(coils, 48, n, 1), bartorch.nufft(coil_images, traj), rtol=1e-4, atol=1e-5
    )


@pytest.mark.skipif(
    not bartorch._cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)
def test_a_bank_left_on_the_host_is_brought_over_a_slab_at_a_time():
    """The card never holds the bank, only the slab being used.

    The loop already reads one slab at a time, so one slab at a time is all
    that has to cross -- which is what lets a bank larger than the card serve
    a reconstruction on it.  The answer has to be the one the card would give
    if it held the whole thing.
    """
    n, coils = 32, 8
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    traj = bt.traj(x=n, y=48, r=True)
    x = bt.phantom([n, n]).reshape(1, n, n)

    resident = linop.NoncartesianSense(maps.cuda(), (coils, n, n), traj=traj.cuda())
    staged = linop.NoncartesianSense(maps, (coils, n, n), traj=traj.cuda())

    # A staged slab is dense where a resident one is a window on to the bank,
    # so the sum that ends the adjoint runs in a different order and the last
    # bit or two of it differ.  What is being checked is the arithmetic, not
    # the order.
    y = resident(x.cuda())
    torch.testing.assert_close(staged(x.cuda()), y, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(staged.adjoint(y), resident.adjoint(y), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        staged.normal(x.cuda()), resident.normal(x.cuda()), rtol=1e-4, atol=1e-5
    )


@pytest.mark.skipif(
    not bartorch._cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)
def test_fetching_a_slab_alongside_the_arithmetic_changes_nothing():
    """With a stream to spare, the next slab is fetched while this one is used.

    BART hands every ``md_`` call the stream of the OpenMP thread that issued
    it, so a region of two puts the fetch on one stream and the arithmetic on
    another.  What that must not change is the answer -- so it is held against
    the same operator driven with one stream, and against one whose bank was
    on the card all along.
    """
    n, coils = 32, 8
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    traj = bt.traj(x=n, y=48, r=True).cuda()
    x = bt.phantom([n, n]).reshape(1, n, n).cuda()

    was = bartorch._cuda.streams()
    try:
        resident = linop.NoncartesianSense(maps.cuda(), (coils, n, n), traj=traj)
        reference = resident.normal(x)

        bartorch._cuda.set_streams(1)
        one = linop.NoncartesianSense(maps, (coils, n, n), traj=traj).normal(x)

        bartorch._cuda.set_streams(2)
        two = linop.NoncartesianSense(maps, (coils, n, n), traj=traj).normal(x)
    finally:
        bartorch._cuda.set_streams(was)

    torch.testing.assert_close(one, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(two, one, rtol=1e-4, atol=1e-5)


def test_a_kernel_bank_serves_a_subspace_operator_as_the_maps_it_stands_for():
    """Kernels and a basis compose.

    A subspace operator carries one volume per coefficient and contracts them
    into frames on the samples' side.  The sensitivities are the same for every
    coefficient, so a bank held as kernels has to apply there as the maps it
    stands for -- and the forward and adjoint have to be each other's adjoint,
    which is what says the coefficients and frames land on the right axes.
    """
    n, spokes, frames, coeffs, coils = 16, 8, 4, 2, 4
    traj = bt.traj(x=n, y=spokes * frames, r=True).reshape(frames, spokes, n, 3)[:, None, None]
    basis = torch.zeros(coeffs, frames, 1, 1, 1, 1, 1, dtype=torch.complex64)
    basis[0, :, 0, 0, 0, 0, 0] = 1.0
    basis[1, :, 0, 0, 0, 0, 0] = torch.linspace(-1, 1, frames)
    kernels, maps = _smooth_bank(n=n, coils=coils)

    dense = linop.NoncartesianSense(maps, (coils, n, n), traj=traj, basis=basis)
    compact = linop.NoncartesianSense(kernels, (coils, n, n), traj=traj, basis=basis, kernels=True)
    assert dense.ishape == (coeffs, 1, 1, 1, 1, n, n)

    torch.manual_seed(0)
    x = torch.randn(dense.ishape, dtype=torch.complex64)
    y = torch.randn(dense.oshape, dtype=torch.complex64)

    torch.testing.assert_close(compact(x), dense(x), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(compact.normal(x), dense.normal(x), rtol=1e-4, atol=1e-5)

    lhs = torch.vdot(dense(x).flatten(), y.flatten())
    rhs = torch.vdot(x.flatten(), dense.adjoint(y).flatten())
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


@pytest.mark.skipif(
    not bartorch._cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)
def test_an_operator_on_a_card_takes_and_returns_host_arrays():
    """The card holds the operator; the caller's arrays stay where they are.

    Built for a card from inputs on the host, the operator brings an image
    over whole once each way and the samples a slab at a time, and answers
    what it answers with everything on the card.  A solver that keeps its
    vectors on the host then uses the card for the operator alone -- and
    between two applications the card holds nothing of the solver's.
    """
    n, spokes, frames, coeffs, coils = 16, 8, 4, 2, 4
    traj = bt.traj(x=n, y=spokes * frames, r=True).reshape(frames, spokes, n, 3)[:, None, None]
    basis = torch.zeros(coeffs, frames, 1, 1, 1, 1, 1, dtype=torch.complex64)
    basis[0, :, 0, 0, 0, 0, 0] = 1.0
    basis[1, :, 0, 0, 0, 0, 0] = torch.linspace(-1, 1, frames)
    kernels, _ = _smooth_bank(n=n, coils=coils)

    on_card = linop.NoncartesianSense(
        kernels.cuda(), (coils, n, n), traj=traj.cuda(), basis=basis.cuda(), kernels=True
    )
    from_host = linop.NoncartesianSense(
        kernels, (coils, n, n), traj=traj, basis=basis, kernels=True, device="cuda"
    )
    assert from_host.device.type == "cuda"

    torch.manual_seed(0)
    x = torch.randn(on_card.ishape, dtype=torch.complex64)
    y = torch.randn(on_card.oshape, dtype=torch.complex64)

    # The adjoint grids with atomic adds, so two of them differ in summation
    # order at about 1e-06; the forward and the normal reproduce exactly.
    for name, got, want, tol in (
        ("forward", from_host(x), on_card(x.cuda()), (1e-5, 1e-6)),
        ("adjoint", from_host.adjoint(y), on_card.adjoint(y.cuda()), (1e-4, 1e-5)),
        ("normal", from_host.normal(x), on_card.normal(x.cuda()), (1e-5, 1e-6)),
    ):
        assert got.device.type == "cpu", name
        torch.testing.assert_close(got, want.cpu(), rtol=tol[0], atol=tol[1], msg=name)

    reused = torch.empty(on_card.ishape, dtype=torch.complex64)
    assert from_host.normal(x, out=reused) is reused
    torch.testing.assert_close(reused, on_card.normal(x.cuda()).cpu(), rtol=1e-5, atol=1e-6)

    cg = optim.CG(maxiter=5)
    solved = cg(y, from_host)
    assert solved.device.type == "cpu"
    torch.testing.assert_close(solved, cg(y.cuda(), on_card).cpu(), rtol=1e-4, atol=1e-5)

    bartorch._cuda.use_memcache(False)
    torch.cuda.synchronize()
    before, _ = torch.cuda.mem_get_info()
    from_host.normal(x)
    torch.cuda.synchronize()
    after, _ = torch.cuda.mem_get_info()
    bartorch._cuda.use_memcache(True)
    assert after >= before - (16 << 20), "a normal left nothing of the caller's on the card"


def test_a_three_dimensional_kernel_bank_applies_as_the_maps_it_stands_for():
    """Inflated an axis at a time, a kernel is the map it stands for in 3D too."""
    n, coils, size = 16, 3, 6
    torch.manual_seed(0)
    kernels = torch.randn(coils, size, size, size, dtype=torch.complex64)
    maps = bartorch.kernels_to_maps(kernels, (n, n, n))
    x = torch.randn(1, n, n, n, dtype=torch.complex64)

    dense = linop.CartesianSense(maps, (coils, n, n, n))
    compact = linop.CartesianSense(kernels, (coils, n, n, n), kernels=True)

    torch.testing.assert_close(compact(x), dense(x), rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not bartorch._cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)
@pytest.mark.parametrize("n, size", [(16, 6), (20, 7), (15, 5)])
def test_a_kernel_bank_inflated_on_a_card_is_the_maps_it_stands_for(n, size):
    """On a card the modulations come out of the loop, and the maps do not change.

    Each axis's centred transform modulates before and after it; the card puts
    the ones before on the kernel and the ones after, with the scale, on the
    map in one pass.  Held against the maps on a grid that is a multiple of
    eight, one that is not, and an odd one, with an odd kernel among them.
    """
    coils = 3
    torch.manual_seed(0)
    kernels = torch.randn(coils, size, size, size, dtype=torch.complex64)
    maps = bartorch.kernels_to_maps(kernels, (n, n, n))
    x = torch.randn(1, n, n, n, dtype=torch.complex64)

    dense = linop.CartesianSense(maps, (coils, n, n, n))
    compact = linop.CartesianSense(kernels.cuda(), (coils, n, n, n), kernels=True)

    y = dense(x)
    torch.testing.assert_close(compact(x.cuda()).cpu(), y, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        compact.adjoint(y.cuda()).cpu(), dense.adjoint(y), rtol=1e-4, atol=1e-5
    )


# --- the coil multiply on its own ---------------------------------------------
#
# The same slab loop with nothing after it, which is what an encoding whose
# transform is not a Fourier transform has to chain its coils onto.


def test_the_coil_multiply_alone_is_barts_own_fmac():
    """With no slab to walk, there is nothing between it and `linop_fmac`."""
    n, coils = 16, 4
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    x = torch.randn(1, n, n, dtype=torch.complex64)

    fmac = linop.MultiplySum(maps.reshape(coils, 1, n, n), (1, n, n), (coils, 1, n, n))
    coil = linop.Coils(maps, (coils, n, n), coil_batch=0)

    assert (coil.ishape, coil.oshape) == (fmac.ishape, fmac.oshape)
    torch.testing.assert_close(coil(x), fmac(x))


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_the_slab_does_not_change_the_coil_multiply(batch):
    n, coils = 16, 4
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    x = torch.randn(1, n, n, dtype=torch.complex64)

    whole = linop.Coils(maps, (coils, n, n), coil_batch=0)
    sliced = linop.Coils(maps, (coils, n, n), coil_batch=batch)

    torch.testing.assert_close(sliced(x), whole(x))
    y = whole(x)
    torch.testing.assert_close(sliced.adjoint(y), whole.adjoint(y))


def test_the_coil_multiply_has_the_adjoint_it_claims():
    n, coils = 16, 3
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    A = linop.Coils(maps, (coils, n, n), coil_batch=2)

    x = torch.randn(*A.ishape, dtype=torch.complex64)
    y = torch.randn(*A.oshape, dtype=torch.complex64)
    lhs = complex((A(x).conj() * y).sum())
    rhs = complex((x.conj() * A.adjoint(y)).sum())
    assert abs(lhs - rhs) < 1e-4 * max(abs(lhs), 1.0)


def test_the_coil_multiply_takes_kernels_as_the_maps_they_stand_for():
    """The point of it: the bank is inflated a slab at a time, never whole."""
    n, coils = 32, 4
    kernels, maps = _smooth_bank(n=n, coils=coils)
    x = bt.phantom([n, n]).reshape(1, n, n)

    dense = linop.Coils(maps, (coils, n, n))
    compact = linop.Coils(kernels, (coils, n, n), kernels=True)

    torch.testing.assert_close(compact(x), dense(x), rtol=1e-4, atol=1e-5)
    y = dense(x)
    torch.testing.assert_close(compact.adjoint(y), dense.adjoint(y), rtol=1e-4, atol=1e-5)


def test_the_coil_multiply_and_an_fft_are_the_cartesian_encoding():
    """Which is what says the slab loop is the transform's to choose.

    `CartesianSense` is the loop with BART's FFT inside it; this is the loop
    with nothing inside it and the FFT chained on afterwards.  The two are the
    same arithmetic in the same order, so they agree exactly rather than
    nearly.
    """
    n, coils = 32, 4
    kernels, maps = _smooth_bank(n=n, coils=coils)
    x = bt.phantom([n, n]).reshape(1, n, n)

    inside = linop.CartesianSense(maps, (coils, n, n))
    outside = linop.FFT(inside.oshape, axes=(-2, -1)) @ linop.Coils(maps, (coils, n, n))

    assert (outside.ishape, outside.oshape) == (inside.ishape, inside.oshape)
    torch.testing.assert_close(outside(x), inside(x), rtol=0, atol=0)


@pytest.mark.parametrize("coils,batch", [(3, 2), (5, 4), (12, 8), (7, 3)])
def test_a_slab_that_does_not_divide_the_coils_is_not_walked_off_the_end(coils, batch):
    """The loop steps by the slab and the transform is built for a slab.

    A last slab with fewer coils in it than that reads sensitivities that are
    not there and writes past the end of the samples, which is heap corruption
    rather than a wrong answer.  The slab is cut down to a divisor of the coil
    count instead, so what changes is residency and never arithmetic.

    Held against the encoding written out rather than against another slab, so
    that a slab which quietly became one coil is still checked against
    something.
    """
    n = 16
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    x = torch.randn(1, n, n, dtype=torch.complex64)

    A = linop.CartesianSense(maps, (coils, n, n), coil_batch=batch)
    want = bartorch.fft(maps.reshape(coils, 1, n, n) * x, axes=(-2, -1), unitary=True)

    torch.testing.assert_close(A(x), want, rtol=1e-5, atol=1e-5)

    y = torch.randn(*A.oshape, dtype=torch.complex64)
    conj = maps.reshape(coils, 1, n, n).conj()
    back = (bartorch.fft(y, axes=(-2, -1), unitary=True, inverse=True) * conj).sum(0)
    torch.testing.assert_close(A.adjoint(y), back, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("coils,batch", [(3, 2), (5, 4), (7, 3)])
def test_an_uneven_slab_is_cut_down_off_the_grid_too(coils, batch):
    n = 16
    torch.manual_seed(0)
    maps = torch.randn(coils, n, n, dtype=torch.complex64)
    traj = bt.traj(x=n, y=24, r=True)
    x = torch.randn(1, n, n, dtype=torch.complex64)

    sliced = linop.NoncartesianSense(maps, (coils, n, n), traj=traj, coil_batch=batch)
    whole = linop.NoncartesianSense(maps, (coils, n, n), traj=traj, coil_batch=0)

    torch.testing.assert_close(sliced(x), whole(x), rtol=1e-4, atol=1e-5)


# --- several sets of maps -----------------------------------------------------
#
# ESPIRiT's second map and ENLIVE's relaxed model give a bank of sets rather
# than one set, and the encoding sums over them: y[c] = sum_m S[m, c] x[m].
# BART contracts that axis itself -- `sense.c` keeps MAPS on the image, drops
# it from the coil images and sums it in `md_ztenmul2` -- so what is added here
# is the shape, not the arithmetic.

SETS = 2


def _sets_bank(n=16, coils=4, sets=SETS, seed=7):
    torch.manual_seed(seed)
    return torch.randn(sets, coils, 1, n, n, dtype=torch.complex64)


def test_a_bank_of_one_set_is_read_either_way():
    """Writing the axis out is allowed and means the same thing."""
    n, coils = 16, 4
    torch.manual_seed(0)
    bare = torch.randn(coils, 1, n, n, dtype=torch.complex64)

    A = linop.Coils(bare, (coils, n, n))
    B = linop.Coils(bare.reshape(1, coils, 1, n, n), (coils, n, n))

    assert A.ishape == B.ishape and A.oshape == B.oshape
    x = torch.randn(*A.ishape, dtype=torch.complex64)
    torch.testing.assert_close(B(x), A(x), rtol=0, atol=0)


def test_a_bank_that_is_neither_shape_is_refused():
    with pytest.raises(ValueError, match="neither .coils, .spatial. nor"):
        linop.Coils(torch.ones(3, 1, 16, 16, dtype=torch.complex64), (4, 16, 16))


def test_the_sets_ride_on_barts_maps_axis():
    n, coils = 16, 4
    A = linop.Coils(_sets_bank(n, coils), (coils, n, n))

    # (coeffs, te, maps, coils, z, y, x): the image carries the sets, the coil
    # images do not, because the operator has summed over them.
    assert A.ishape == (1, 1, SETS, 1, 1, n, n)
    assert A.oshape == (1, 1, 1, coils, 1, n, n)


def test_several_sets_are_summed_the_way_enlive_means_it():
    n, coils = 16, 4
    bank = _sets_bank(n, coils)
    A = linop.Coils(bank, (coils, n, n))

    torch.manual_seed(0)
    x = torch.randn(*A.ishape, dtype=torch.complex64)
    want = (bank.reshape(1, 1, SETS, coils, 1, n, n) * x.reshape(1, 1, SETS, 1, 1, n, n)).sum(
        2, keepdim=True
    )
    torch.testing.assert_close(A(x), want, rtol=0, atol=0)


def test_several_sets_have_the_adjoint_they_claim():
    n, coils = 16, 4
    A = linop.Coils(_sets_bank(n, coils), (coils, n, n), coil_batch=2)

    torch.manual_seed(0)
    x = torch.randn(*A.ishape, dtype=torch.complex64)
    y = torch.randn(*A.oshape, dtype=torch.complex64)
    lhs = complex((A(x).conj() * y).sum())
    rhs = complex((x.conj() * A.adjoint(y)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-5


@pytest.mark.parametrize("batch", [0, 1, 2, 4])
def test_the_slab_walks_the_coils_and_not_the_sets(batch):
    """The slab is a slab of coils; the sets are inside it whatever it is."""
    n, coils = 16, 4
    bank = _sets_bank(n, coils)
    sliced = linop.Coils(bank, (coils, n, n), coil_batch=batch)
    whole = linop.Coils(bank, (coils, n, n), coil_batch=0)

    torch.manual_seed(0)
    x = torch.randn(*sliced.ishape, dtype=torch.complex64)
    torch.testing.assert_close(sliced(x), whole(x), rtol=1e-6, atol=1e-6)


def test_several_sets_held_as_kernels():
    n, coils = 16, 4
    torch.manual_seed(8)
    kernels = torch.randn(SETS, coils, 1, 5, 5, dtype=torch.complex64)
    dense = bartorch.kernels_to_maps(kernels, (1, n, n))

    a = linop.Coils(dense, (coils, n, n))
    b = linop.Coils(kernels, (coils, n, n), kernels=True)

    torch.manual_seed(0)
    x = torch.randn(*a.ishape, dtype=torch.complex64)
    torch.testing.assert_close(b(x), a(x), rtol=1e-4, atol=1e-5)


def test_several_sets_off_the_grid_are_the_sum_of_the_one_set_operators():
    """Which is what summing over the maps axis means, said another way."""
    n, coils = 16, 4
    bank = _sets_bank(n, coils)
    traj = bt.traj(x=n, y=24, r=True)

    A = linop.NoncartesianSense(bank, (coils, n, n), traj=traj)
    torch.manual_seed(0)
    x = torch.randn(*A.ishape, dtype=torch.complex64)

    want = sum(
        linop.NoncartesianSense(bank[m], (coils, n, n), traj=traj)(x[0, 0, m, 0])
        for m in range(SETS)
    )
    torch.testing.assert_close(A(x).reshape(want.shape), want, rtol=1e-4, atol=1e-5)
