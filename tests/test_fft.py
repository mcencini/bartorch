"""BART's FFT, and the two transforms that serve it.

BART plans through the FFTW guru interface: a set of transformed dimensions
and a set of dimensions to loop over, each with its own strides.  MKL takes
the transformed axes and one loop axis, and the loop axes left over are walked
by the library.  That walk is the part worth testing, because a wrong stride
in it is not a crash but a quietly rearranged image.

The two implementations are held against each other and both against numpy, so
neither a shared convention nor a shared mistake can pass.
"""

import numpy as np
import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _backend

requires_mkl_fft = pytest.mark.skipif(
    bartorch.backend_sources().get("fft", "built-in") == "built-in",
    reason="no MKL in this process, so there is only one transform to test",
)


def _centred_reference(x: torch.Tensor, axes: tuple[int, ...]) -> np.ndarray:
    """What BART's ``fft`` computes, in numpy: centred, unnormalised."""
    a = x.numpy()
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a, axes=axes), axes=axes), axes=axes)


# Shapes chosen for what they leave BART to loop over once the transformed
# axes are taken out: nothing, one axis, and two -- the last being the case
# MKL cannot describe by itself.
CASES = [
    ((16, 24), (-2, -1)),
    ((4, 16, 24), (-2, -1)),
    ((2, 3, 16, 24), (-2, -1)),
    ((2, 3, 8, 10, 12), (-3, -2, -1)),
    ((5, 7, 12), (-1,)),
]


@pytest.mark.parametrize("shape,axes", CASES)
def test_the_fft_matches_numpy_however_many_axes_are_looped_over(shape, axes):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.complex64)
    got = bartorch.fft(x, axes=axes).numpy()
    ref = _centred_reference(x, axes)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4 * float(np.abs(ref).max()))


@requires_mkl_fft
@pytest.mark.parametrize("shape,axes", CASES)
def test_mkl_and_the_built_in_transform_agree(shape, axes):
    """The same description, given to two libraries, has to mean the same thing."""
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.complex64)

    with_mkl = bartorch.fft(x, axes=axes)
    with _backend.built_in_fft():
        built_in = bartorch.fft(x, axes=axes)

    torch.testing.assert_close(with_mkl, built_in, rtol=1e-4, atol=1e-4)


@requires_mkl_fft
def test_an_in_place_transform_agrees_too():
    """BART transforms a point spread function in place, over a doubled grid.

    In place and out of place are different descriptors, so this reaches code
    the out-of-place cases do not.
    """
    n = 32
    traj = bt.traj(x=n, y=48, r=True)

    with_mkl = bt.psf(traj)
    with _backend.built_in_fft():
        built_in = bt.psf(traj)

    scale = float(built_in.abs().max())
    assert float((with_mkl - built_in).abs().max()) / scale < 1e-4


@requires_mkl_fft
def test_a_reconstruction_agrees_whichever_transform_serves_it():
    n, coils = 32, 2
    torch.manual_seed(0)
    traj = bt.traj(x=n, y=48, r=True)
    image = bt.phantom([n, n], coils=coils)
    ksp = bartorch.nufft(image, traj)
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    with_mkl = bt.pics(ksp, maps, t=traj)
    with _backend.built_in_fft():
        built_in = bt.pics(ksp, maps, t=traj)

    scale = float(built_in.abs().max())
    assert float((with_mkl - built_in).abs().max()) / scale < 1e-3


@requires_mkl_fft
def test_every_plan_a_reconstruction_makes_goes_to_mkl():
    """A description MKL declines is served, but silently, so it is counted."""
    from bartorch._lib import library

    lib = library()
    n, coils = 32, 2
    traj = bt.traj(x=n, y=48, r=True)
    ksp = bartorch.nufft(bt.phantom([n, n], coils=coils), traj)
    maps = torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5

    lib.bartorch_fft_reset_counters()
    bt.pics(ksp, maps, t=traj)
    bt.nlinv(ksp, t=traj, maxiter=3)

    by_mkl, by_built_in = lib.bartorch_fft_counter(0), lib.bartorch_fft_counter(1)
    assert by_mkl > 0
    assert by_built_in == 0, f"{by_built_in} of {by_mkl + by_built_in} plans were declined"


def test_the_fft_reports_where_it_came_from():
    source = bartorch.backend_sources().get("fft")
    assert source, "the FFT was never resolved"
    assert source == "built-in" or "mkl" in source.lower() or "torch" in source.lower(), source
