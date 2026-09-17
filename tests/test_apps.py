"""``bartorch.apps`` against the applications they are assembled from.

An app is a re-expression of a BART application, so on a grid the test is
``torch.equal``: a difference in the last place would mean some step was done
twice, once by BART and once by this package.
"""

from __future__ import annotations

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _dispatch, _finufft, apps, linop, priors

SIZE, COILS, ACCEL = 24, 4, 2


@pytest.fixture
def _whole_coil_operator():
    """``pics`` walks every coil at once; the package's default is a slab."""
    before = _dispatch.coil_batch()
    _dispatch.set_coil_batch(0)
    yield
    _dispatch.set_coil_batch(before)


def _cartesian():
    kspace = bt.phantom(SIZE, coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    mask = torch.zeros(SIZE, dtype=torch.complex64)
    mask[::ACCEL] = 1
    mask[SIZE // 2 - 2 : SIZE // 2 + 2] = 1
    return kspace * mask.reshape(SIZE, 1), maps


def _wavelet(**kwargs):
    return priors.Wavelet(axes=(-1, -2), weight=0.01, **kwargs)


def _tv(weight=0.01):
    return priors.TotalVariation(axes=(-1, -2), weight=weight)


def _llr():
    return priors.LocallyLowRank(axes=(-1, -2), weight=0.01, block=4)


def _tgv(weight=0.01):
    return priors.TotalGeneralizedVariation(axes=(-1, -2), weight=weight)


#: What the tool is given.  The app is given the same, which is the point: the
#: two take the same arguments and the app is the one without the command.
_CONFIGURATIONS = [
    ("plain", {}),
    ("tikhonov", {"l2": 0.1}),
    ("wavelet admm", {"regularizers": _wavelet(), "solver": "admm"}),
    ("wavelet fista", {"regularizers": _wavelet(), "solver": "fista"}),
    ("wavelet ist", {"regularizers": _wavelet(), "solver": "ist"}),
    ("no cycle spinning", {"regularizers": _wavelet(randshift=False), "solver": "fista"}),
    ("tv pridu", {"regularizers": _tv(), "solver": "pridu"}),
    ("tv admm", {"regularizers": _tv(0.005), "solver": "admm"}),
    ("locally low rank", {"regularizers": _llr(), "solver": "admm"}),
    ("two terms", {"regularizers": [_wavelet(), _tv(0.005)], "solver": "admm"}),
    ("tgv admm", {"regularizers": _tgv(), "solver": "admm"}),
    ("tgv pridu", {"regularizers": _tgv(), "solver": "pridu"}),
    # No solver named: the app chooses the one `italgo_choose` chooses.
    ("wavelet, chosen", {"regularizers": _wavelet()}),
    ("tv, chosen", {"regularizers": _tv()}),
    ("two terms, chosen", {"regularizers": [_wavelet(), _tv(0.005)]}),
]


@pytest.mark.parametrize(
    "arguments", [a for _, a in _CONFIGURATIONS], ids=[n for n, _ in _CONFIGURATIONS]
)
def test_the_app_is_the_tool_to_the_last_bit(arguments, _whole_coil_operator):
    kspace, maps = _cartesian()
    tool = bt.pics(kspace, maps, maxiter=20, **arguments).squeeze()
    ours = apps.pics(kspace, maps, maxiter=20, **arguments).squeeze()
    assert torch.equal(ours, tool), f"maximum difference {float((ours - tool).abs().max()):.3e}"


def test_the_iteration_is_the_one_the_terms_choose():
    """``italgo_choose`` (grecon/italgo.c), written out: an l2 penalty on the
    image leaves the choice where it was, the total variations take ADMM, and
    anything else takes FISTA first and ADMM after."""
    from bartorch.apps.pics import _chosen

    assert _chosen([]) == "cg"
    assert _chosen([priors.L2(0.01)]) == "cg"
    assert _chosen([_wavelet()]) == "fista"
    assert _chosen([_tv()]) == "admm"
    assert _chosen([_wavelet(), _tv()]) == "admm"
    assert _chosen([_tv(), _wavelet()]) == "admm"
    assert _chosen([priors.ImageNIHT((-1, -2), 4)]) == "niht"


def test_an_unknown_solver_is_refused():
    kspace, maps = _cartesian()
    with pytest.raises(ValueError, match="solver must be one of"):
        apps.pics(kspace, maps, solver="newton")


def _radial(size=32, coils=4, spokes=32):
    sens = bt.coils(t=bt.grid(D=(size, size, 1)), n=coils)[:, 0]
    sens = sens / bartorch.rss(sens, axes=(0,), keepdim=True)
    traj = bt.traj(readout=size, spokes=spokes, radial=True, golden=True)
    encoding = linop.NoncartesianSense(sens, (size, size), traj=traj)
    image = torch.as_tensor(bt.phantom(size)).to(torch.complex64)
    return traj, sens[:, None], bt.noise(encoding(image), n=1e-6, s=7)


def test_off_the_grid_the_app_is_the_tool_to_round_off():
    """Not the bits, and they cannot be: a non-uniform transform spread over
    threads sums in the order the threads finish in, so nothing off the grid is
    bit-reproducible.  ``pics`` is not reproducible against itself there --
    twice over the same data it differs by about 6e-07 of the peak, the same
    size as the difference asserted here -- so an equality would be a statement
    about the thread count rather than about the pipeline.
    """
    traj, maps, measured = _radial()
    term = _tv(0.001)
    tool = bt.pics(
        measured[..., None], maps, traj=traj, regularizers=term, solver="admm", maxiter=10
    ).squeeze()
    ours = apps.pics(
        measured, maps, traj=traj, regularizers=term, solver="admm", maxiter=10
    ).squeeze()
    assert float((ours - tool).abs().max()) < 1e-5 * float(tool.abs().max())
