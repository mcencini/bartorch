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


@pytest.mark.parametrize("solver", ["cg", "ist", "fista", "admm"])
def test_a_warm_start_reaches_the_iteration(solver, _whole_coil_operator):
    """And moves the answer, so the equality is not two cold starts agreeing.

    ``pics -W`` rescales the warm start only under ``-S``, where the answer is
    put back into the data's units at the end (pics.c:592).  Without it the
    solve stays in the scaled units and so does the start, which is the way
    the app takes one.
    """
    kspace, maps = _cartesian()
    arguments = {} if solver == "cg" else {"regularizers": _wavelet(), "solver": solver}
    warm = 0.5 * apps.pics(kspace, maps, maxiter=5)

    tool = bt.pics(kspace, maps, maxiter=20, W=warm, **arguments).squeeze()
    ours = apps.pics(kspace, maps, maxiter=20, initial=warm, **arguments).squeeze()
    cold = bt.pics(kspace, maps, maxiter=20, **arguments).squeeze()

    assert torch.equal(ours, tool)
    assert not torch.equal(tool, cold), "the warm start changed nothing, so this proves nothing"


def test_the_eigenvalue_step_is_the_tools():
    """``pics -e`` is ``eigen ? 30 : 0`` power iterations (pics.c:667).

    Held to round-off rather than to the bits: the starting vector comes from
    BART's process-global generator, so the tool's call and the app's do not
    start it from the same place.
    """
    kspace, maps = _cartesian()
    arguments = {"regularizers": _wavelet(), "solver": "fista"}
    tool = bt.pics(kspace, maps, maxiter=20, eigen_step=True, **arguments).squeeze()
    ours = apps.pics(kspace, maps, maxiter=20, eigen_step=True, **arguments).squeeze()
    plain = apps.pics(kspace, maps, maxiter=20, **arguments).squeeze()

    assert float((ours - tool).abs().max()) < 1e-5 * float(tool.abs().max())
    assert not torch.equal(ours, plain), "the eigenvalue step changed nothing"


def test_conjugate_gradients_takes_no_eigenvalue_step():
    kspace, maps = _cartesian()
    with pytest.raises(ValueError, match="cg does not take"):
        apps.pics(kspace, maps, eigen_step=True)


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


# --- mobafit ---------------------------------------------------------------
#
# The model the app fits is TorchSim's rather than BART's, so there is nothing
# to be equal to; what a fit has to answer for is the relaxation time the data
# was made from, and the decay and the recovery are written out here.

FIT_SIZE = 12
ECHO_TIMES = [12.5 * (echo + 1) for echo in range(8)]
INVERSION_TIMES = [50.0, 150.0, 400.0, 900.0, 1500.0, 2500.0, 4000.0]


def _two_halves(left: float, right: float, size: int = FIT_SIZE) -> torch.Tensor:
    """A map of two constant halves, so one fit answers two relaxation times."""
    values = torch.full((size, size), left)
    values[:, size // 2 :] = right
    return values


def test_a_decay_is_fitted_to_the_time_it_was_made_from():
    """``M0 exp(-TE / T2)``, written out here and recovered from the images."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)

    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    fitted = apps.mobafit(images, model, T2=80.0)["T2"]

    assert torch.allclose(fitted, t2, rtol=1e-3)


def test_a_recovery_is_fitted_to_the_time_it_was_made_from():
    from bartorch import nlop

    model = nlop.InversionRecovery(INVERSION_TIMES, (FIT_SIZE, FIT_SIZE))
    left = model(model.initial(T1=800.0))
    right = model(model.initial(T1=1400.0))
    images = torch.cat([left[..., : FIT_SIZE // 2], right[..., FIT_SIZE // 2 :]], dim=-1)

    fitted = apps.mobafit(images, model, T1=1000.0)["T1"]

    assert torch.allclose(fitted, _two_halves(800.0, 1400.0), rtol=1e-3)


def test_the_magnitude_is_fitted_where_the_phase_is_thrown_away():
    """``mobafit -a``: a decay whose phase varies across the image, given to
    the fit as a magnitude, is the same T2 as the decay itself."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    phase = torch.exp(1j * torch.linspace(-1.0, 1.0, FIT_SIZE))[None, :]
    images = (torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2) * phase).abs()

    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    fitted = apps.mobafit(images.to(torch.complex64), model, magnitude=True, T2=80.0)["T2"]

    assert torch.allclose(fitted, t2, rtol=1e-3)


def test_a_voxel_with_no_signal_keeps_the_value_it_started_from():
    """``mobafit`` skips a patch whose data is zero; an unconstrained voxel
    would otherwise walk wherever the bounds allow."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    images[:, 0] = 0.0

    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))
    fitted = apps.mobafit(images, model, T2=80.0)["T2"]

    assert torch.allclose(fitted[0], torch.full((FIT_SIZE,), 80.0), atol=1e-3)
    assert torch.allclose(fitted[1:], t2[1:], rtol=1e-3)


def test_the_fit_is_taken_from_where_it_is_started():
    """The starting maps reach the loop: a fit stopped after one step is still
    near where it began, and a different beginning is a different answer."""
    from bartorch import nlop

    t2 = _two_halves(60.0, 110.0)
    images = torch.exp(-torch.tensor(ECHO_TIMES)[:, None, None] / t2).to(torch.complex64)
    model = nlop.MultiEcho(ECHO_TIMES, (FIT_SIZE, FIT_SIZE))

    one = apps.mobafit(images, model, iterations=1, T2=40.0)["T2"]
    other = apps.mobafit(images, model, iterations=1, T2=200.0)["T2"]

    assert float((one - other).abs().max()) > 1.0
    assert float(one.median()) < float(other.median())


# --- pocsense --------------------------------------------------------------

#: What the application is given, and what the app is given for the same run.
_POCS_CONFIGURATIONS = [
    ("plain", {}, {}),
    ("l2", {"r": 0.1}, {"alpha": 0.1}),
    ("wavelet", {"r": 0.01, "l": 1}, {"alpha": 0.01, "wavelet": True}),
    ("robust", {"o": 0.05}, {"robust": 0.05}),
    (
        "robust wavelet",
        {"o": 0.05, "r": 0.01, "l": 1},
        {"robust": 0.05, "alpha": 0.01, "wavelet": True},
    ),
    ("one sweep", {"i": 1}, {"maxiter": 1}),
]


@pytest.mark.parametrize(
    "arguments",
    [(t, a) for _, t, a in _POCS_CONFIGURATIONS],
    ids=[n for n, _, _ in _POCS_CONFIGURATIONS],
)
def test_the_pocsense_app_is_the_tool_to_the_last_bit(arguments):
    flags, ours = arguments
    kspace, maps = _cartesian()
    tool = bt.pocsense(kspace, maps, **{"i": 10, **flags})
    made = apps.pocsense(kspace, maps, **{"maxiter": 10, **ours})
    assert torch.equal(made, tool), f"maximum difference {float((made - tool).abs().max()):.3e}"


def test_an_odd_grid_is_the_tool_too():
    """The modulation is a phase rather than a sign on an odd axis, so which
    two of the three factors of the sparsity projection are multiplied first
    is visible in the last bits."""
    kspace = bt.phantom(25, coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    mask = torch.zeros(25, dtype=torch.complex64)
    mask[::ACCEL] = 1
    mask[25 // 2 - 2 : 25 // 2 + 2] = 1
    kspace = kspace * mask.reshape(25, 1)

    tool = bt.pocsense(kspace, maps, i=10, r=0.01, l=1)
    ours = apps.pocsense(kspace, maps, maxiter=10, alpha=0.01, wavelet=True)
    assert torch.equal(ours, tool)


def test_a_volume_is_the_tool_too():
    """Three transformed axes rather than two, which is what the app works
    out from the sensitivities rather than assuming."""
    size = 16
    kspace = bt.phantom([size, size, size], coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    mask = torch.zeros(size, size, 1, dtype=torch.complex64)
    mask[torch.randperm(size, generator=torch.Generator().manual_seed(3))[: size // 2]] = 1
    mask[size // 2 - 1 : size // 2 + 1] = 1
    kspace = kspace * mask

    tool = bt.pocsense(kspace, maps, i=5, r=0.01, l=1)
    ours = apps.pocsense(kspace, maps, maxiter=5, alpha=0.01, wavelet=True)
    assert torch.equal(ours, tool)


def test_the_sweep_is_the_projections_in_order():
    """``pocs`` applies every projection in turn and keeps nothing else, so a
    sweep is the composition and a run is that composition repeated."""
    from bartorch import optim

    first = torch.zeros(4, dtype=torch.complex64)
    steps: list[str] = []

    def one(x):
        steps.append("one")
        return x + 1.0

    def half(x):
        steps.append("half")
        return x * 0.5

    answer = optim.POCS([one, half], maxiter=3)(first)

    assert steps == ["one", "half", "one", "half", "one", "half"]
    # x -> (x + 1) / 2 from zero: 1/2, 3/4, 7/8.
    assert torch.allclose(answer, torch.full((4,), 0.875, dtype=torch.complex64))


def test_a_sweep_starts_where_it_is_told():
    from bartorch import optim

    block = optim.POCSBlock([lambda x: x * 2.0])
    y = torch.ones(3, dtype=torch.complex64)

    assert torch.equal(block.start(y).x, torch.zeros(3, dtype=torch.complex64))
    assert torch.equal(block.start(y, None, y).x, y)
    assert torch.equal(block.output(block(block.start(y, None, y))), 2.0 * y)


def test_a_term_is_taken_as_its_projection():
    """``pocs`` calls a projection with ``mu = 1``, so a term enters as its
    proximal operator at that step size."""
    from bartorch import optim, priors

    x = torch.tensor([4.0, -2.0, 0.5], dtype=torch.complex64)
    term = priors.L2(0.25)

    swept = optim.POCS([term], maxiter=1)(x, None, x)

    assert torch.allclose(swept, x / 1.25)


def test_a_sweep_of_nothing_is_refused():
    from bartorch import optim

    with pytest.raises(ValueError, match="no projections"):
        optim.POCS([], maxiter=1)(torch.zeros(2, dtype=torch.complex64))


def test_a_projection_has_to_be_one():
    from bartorch import optim

    with pytest.raises(TypeError, match="callable on a tensor"):
        optim.POCSBlock([3])
