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
from bartorch._lib import library


@pytest.fixture
def restore_batch():
    was = bartorch.coil_batch()
    yield
    bartorch.set_coil_batch(was)


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
    kspace = bt.fft(image, axes=(-2, -1))

    bartorch.set_coil_batch(0)
    reference = bt.pics(kspace, maps, iter_=30)

    bartorch.set_coil_batch(batch)
    torch.testing.assert_close(bt.pics(kspace, maps, iter_=30), reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch", BATCHES)
def test_a_non_cartesian_reconstruction_is_the_same_whatever_the_slab(batch, restore_batch):
    image, maps = _problem()
    traj = bt.traj(x=32, y=48, r=True)
    kspace = bt.nufft(traj, image)

    bartorch.set_coil_batch(0)
    reference = bt.pics(kspace, maps, t=traj, iter_=30)

    bartorch.set_coil_batch(batch)
    got = bt.pics(kspace, maps, t=traj, iter_=30)

    scale = float(reference.abs().max())
    assert float((got - reference).abs().max()) / scale < 1e-4


def test_the_slab_is_actually_taken(restore_batch):
    """A slab that silently fell back to BART's chain would pass every test above."""
    lib = library()
    image, maps = _problem()
    traj = bt.traj(x=32, y=48, r=True)
    kspace = bt.nufft(traj, image)

    bartorch.set_coil_batch(0)
    lib.bartorch_sense_reset_counters()
    bt.pics(kspace, maps, t=traj, iter_=3)
    assert (lib.bartorch_sense_counter(0), lib.bartorch_sense_counter(1)) == (0, 1)

    bartorch.set_coil_batch(2)
    lib.bartorch_sense_reset_counters()
    bt.pics(kspace, maps, t=traj, iter_=3)
    bt.pics(bt.fft(image, axes=(-2, -1)), maps, iter_=3)
    assert lib.bartorch_sense_counter(0) == 2, "both SENSE operators walk their coils"
    assert lib.bartorch_sense_counter(1) == 0


def test_a_single_coil_goes_back_to_barts_own_operator(restore_batch):
    """There is nothing to slab, and an operator that pretended otherwise
    would pay for a loop of one."""
    lib = library()
    n = 32
    image = bt.phantom([n, n]).reshape(1, 1, n, n)
    maps = torch.ones(1, 1, 1, n, n, dtype=torch.complex64)

    bartorch.set_coil_batch(1)
    lib.bartorch_sense_reset_counters()
    bt.pics(bt.fft(image, axes=(-2, -1)), maps, iter_=3)
    assert lib.bartorch_sense_counter(0) == 0
    assert lib.bartorch_sense_counter(1) == 1


def test_the_setting_reports_itself(restore_batch):
    bartorch.set_coil_batch(3)
    assert bartorch.coil_batch() == 3
    bartorch.set_coil_batch(0)
    assert bartorch.coil_batch() == 0
