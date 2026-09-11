"""The solvers in ``bartorch.optim``, which are BART's and not this package's.

``bartorch_solve`` hands BART's ``lsqr2`` the encoding, the proximal operators
and the iteration ``italgo_config`` makes, which is what ``pics`` calls.

Two things are checked.  That the mechanism is BART's: the loop does not cross
back into Python, every iteration is reachable, a term means what its ``-R``
string means, and nothing is rebuilt per solve.  And that a reconstruction
assembled from operators, terms and a solver is ``bart pics`` to the bit,
across nine configurations (``test_an_assembled_pics_is_the_tool_to_the_last_bit``).
"""

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _dispatch, linop, optim, prox


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _unitary():
    """An operator whose normal is the identity, so the answer is known."""
    return linop.FFT((8, 8), axes=(-1, -2))


# --- the loop stays in C ----------------------------------------------------


def test_a_bart_operator_is_handed_over_as_itself():
    """An encoding BART built is given to BART's solver as it stands, so the
    iteration has nothing to call back into."""
    A = _unitary()
    assert A._bart() is A


def test_an_operator_written_here_is_the_one_that_costs_a_crossing():
    """Its callbacks fire during the solve: one crossing per application."""
    A = _unitary()
    calls = []

    def forward(x):
        calls.append("forward")
        return A.forward(x)

    def adjoint(x):
        calls.append("adjoint")
        return A.adjoint(x)

    P = linop.Callback((8, 8), (8, 8), forward, adjoint)
    optim.CG(maxiter=5)(_rand(8, 8), P)
    assert calls, "a Python operator was not called at all"


def test_the_solve_is_one_call_into_the_library():
    """However many iterations it runs."""
    from bartorch._lib import library

    A = _unitary()
    y = _rand(8, 8)
    lib = library()
    entered = []
    original = lib.bartorch_solve

    class Counting:
        def __call__(self, *args):
            entered.append(1)
            return original(*args)

    lib.bartorch_solve = Counting()
    try:
        optim.CG(maxiter=50)(y, A)
    finally:
        lib.bartorch_solve = original
    assert entered == [1]


def test_the_solve_builds_no_operator_of_its_own():
    """BART counts every transform it builds, so a solve that built its own
    copy of the encoding would show up here."""
    from bartorch import _finufft as finufft

    n = 16
    traj = bt.traj(readout=n, spokes=24, radial=True)
    maps = _rand(2, n, n)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    A = linop.Sense(maps, (2, n, n), traj=traj)
    y = A(_rand(1, n, n))
    term = prox.Wavelet(axes=(-1, -2), weight=0.01)
    term.build(A.ishape)

    finufft.reset_counters()
    optim.FISTA(term, maxiter=25, eigen=True)(y, A)
    assert finufft.operators_built() == (0, 0)


# --- it is BART's iteration -------------------------------------------------


def test_the_tolerance_stops_conjugate_gradients_early():
    """``italgo_config`` leaves the tolerance at zero; the solve sets it."""
    diag = torch.linspace(0.1, 1.0, 64).to(torch.complex64).reshape(8, 8)
    A = linop.Diagonal(diag, (8, 8))
    truth = _rand(8, 8)
    full = optim.CG(maxiter=64)(A(truth), A)
    early = optim.CG(maxiter=64, tol=0.5)(A(truth), A)
    torch.testing.assert_close(full, truth, rtol=1e-3, atol=1e-3)
    assert not torch.equal(early, full)


def test_an_orthonormal_encoding_gives_back_what_it_was_given():
    A = _unitary()
    truth = _rand(8, 8)
    torch.testing.assert_close(optim.CG(maxiter=40)(A(truth), A), truth, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize(
    "make",
    [
        lambda term: optim.CG(maxiter=5),
        lambda term: optim.IST(term, maxiter=5),
        lambda term: optim.FISTA(term, maxiter=5),
        lambda term: optim.ADMM(term, maxiter=5),
        lambda term: optim.PRIDU(term, maxiter=5),
    ],
    ids=["cg", "ist", "fista", "admm", "pridu"],
)
def test_every_iteration_bart_has_is_reachable(make):
    A = _unitary()
    y = _rand(8, 8)
    got = make(prox.Wavelet(axes=(-1, -2), weight=0.01))(y, A)
    assert got.shape == A.ishape
    assert torch.isfinite(got.abs()).all()


def test_a_heavier_weight_shrinks_the_answer():
    A = _unitary()
    y = _rand(8, 8)
    # This operator's normal is the identity, so one is the right step.
    light = optim.FISTA(prox.Wavelet(axes=(-1, -2), weight=0.001), maxiter=30, step=1.0)(y, A)
    heavy = optim.FISTA(prox.Wavelet(axes=(-1, -2), weight=0.5), maxiter=30, step=1.0)(y, A)
    assert heavy.abs().sum() < light.abs().sum()


def test_several_terms_are_taken_together():
    terms = [
        prox.Wavelet(axes=(-1, -2), weight=0.01),
        prox.TotalVariation(axes=(-1, -2), weight=0.01),
    ]
    got = optim.ADMM(terms, maxiter=5)(_rand(8, 8), _unitary())
    assert torch.isfinite(got.abs()).all()


@pytest.mark.parametrize(
    "term",
    [
        prox.Wavelet(axes=(-1, -2), weight=0.01),
        prox.TotalVariation(axes=(-1, -2), weight=0.01),
        prox.LocallyLowRank(axes=(-1, -2), weight=0.01),
        prox.Laplace(axes=(-1, -2), weight=0.01),
        prox.L1(weight=0.01),
        prox.L2(weight=0.1),
        prox.NonNegative(),
        prox.ImaginaryL1(weight=0.01),
    ],
    ids=lambda t: type(t).__name__,
)
def test_every_term_bart_has_can_be_asked_for(term):
    got = optim.ADMM(term, maxiter=5)(_rand(8, 8), _unitary())
    assert torch.isfinite(got.abs()).all()


def test_a_term_is_built_once_and_handed_over_as_it_stands():
    """The object owns what BART made of it, so a second solve builds nothing."""
    term = prox.Wavelet(axes=(-1, -2), weight=0.01)
    first = term.build((8, 8))
    assert term.build((8, 8)) == first

    A = _unitary()
    y = _rand(8, 8)
    fista = optim.FISTA(term, maxiter=20, step=1.0)
    once = fista(y, A)
    twice = fista(y, A)
    assert torch.equal(once, twice)
    assert term.build((8, 8)) == first


@pytest.mark.parametrize(
    "term",
    [
        prox.TotalGeneralizedVariation((-1, -2), 0.01),
        prox.InfimalConvolutionTV((-1, -2), 0.01),
        prox.InfimalConvolutionTGV((-1, -2), 0.01),
    ],
    ids=repr,
)
def test_a_term_bart_configures_with_the_whole_set_is_left_to_pics(term):
    """Total generalized variation and the infimal convolutions extend the
    optimization variable, which BART counts across every term, so one cannot
    be built alone.  ``tools.pics`` takes them."""
    with pytest.raises(TypeError, match="tools.pics"):
        optim.ADMM(term)


@pytest.mark.parametrize(
    "term,ndim,string",
    [
        (prox.Wavelet(axes=(-1, -2), weight=0.01), 2, "W:3:0:0.01"),
        (prox.Wavelet(axes=(-1, -2), weight=0.01, joint_axes=0), 3, "W:3:4:0.01"),
        (prox.Wavelet(axes=0, weight=0.01), 3, "W:4:0:0.01"),
        (prox.L1(0.02, joint_axes=-3), 3, "I:4:0.02"),
        (prox.L2(0.5), 2, "Q:0.5"),
        (prox.NonNegative(), 2, "S"),
        (prox.WaveletNIHT(axes=(-1, -2), count=10), 2, "H:3:0:10"),
        (prox.TotalGeneralizedVariation((-1, -2), 0.01), 2, "G:3:0:0.01"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_term_is_the_string_the_parser_reads(term, ndim, string):
    """``-R W:3:0:0.01`` is a letter, two bitmasks and a weight, and so is the
    object, with the axes written as axes."""
    assert term._argument(ndim) == string


def test_an_axis_is_an_axis_and_not_a_bitmask():
    term = prox.Wavelet(axes=(-1, -2), weight=0.01)
    assert term._flags(ndim=2) == (3, 0)
    assert term._flags(ndim=3) == (3, 0)
    assert prox.Wavelet(axes=0, weight=0.01)._flags(ndim=3) == (4, 0)


def test_a_string_says_what_to_use_instead():
    with pytest.raises(TypeError, match="prox.Wavelet"):
        optim.FISTA("W:3:0:0.01")


# --- what it refuses --------------------------------------------------------


def test_niht_takes_only_hard_thresholding_terms():
    with pytest.raises(TypeError, match="NIHT"):
        optim.NIHT(prox.Wavelet(axes=(-1, -2), weight=0.01))


def test_something_that_is_not_a_term_is_refused():
    with pytest.raises(TypeError, match="bartorch.prox"):
        optim.ADMM(object())


def test_a_wavelet_family_bart_does_not_have_is_refused_here():
    """``opt_reg_configure`` answers an unknown family with ``error()``."""
    with pytest.raises(ValueError, match="family"):
        prox.Wavelet(axes=(-1, -2), weight=0.01, family="db4")


def test_the_terms_are_the_ones_barts_parser_knows():
    """Every letter offered here is one ``grecon/optreg.c`` reads, so a term
    cannot be asked for that BART would answer with ``error()`` -- which leaves
    the next call into the library spinning."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = root / "external" / "bart" / "src" / "grecon" / "optreg.c"
    if not source.exists():
        pytest.skip("the BART submodule is not checked out")
    bart_knows = set(re.findall(r'strcmp\(rt, "([A-Za-z0-9]+)"\)', source.read_text()))
    offered = {
        getattr(prox, name).kind
        for name in prox.__all__
        if isinstance(getattr(prox, name), type) and getattr(prox, name).kind
    }
    assert offered <= bart_knows


# --- against the tool -------------------------------------------------------


@pytest.fixture
def _whole_coil_operator():
    """BART's own SENSE operator, which is what the tool builds.

    The coil-slab operator gives the same answer by different arithmetic, so
    the bit-for-bit comparison is made against BART's.  Process-wide, hence
    put back afterwards.
    """
    before = _dispatch.coil_batch()
    _dispatch.set_coil_batch(0)
    yield
    _dispatch.set_coil_batch(before)


def _pics_problem(size=24, coils=4, accel=2):
    """A ``pics`` problem, and the same problem assembled from this package.

    ``pics`` applies the sampling pattern to the k-space (pics.c:425), the
    modulation that moves the FFT's centre (pics.c:437), and the scaling it
    estimates from what is left (pics.c:501) before it iterates; the assembly
    does the same.  The encoding is the one ``grecon/model.c`` builds: SENSE
    with the sampling chained on.
    """
    kspace = bt.phantom(size, coils=coils, kspace=True)
    maps = bt.ecalib(kspace, maps=1)

    mask = torch.zeros(size, dtype=torch.complex64)
    mask[::accel] = 1
    mask[size // 2 - 2 : size // 2 + 2] = 1
    kspace = kspace * mask.reshape(size, 1)

    pattern = bt.pattern(kspace)
    y = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
    scale = optim.data_scaling(y)

    S = linop.Sense(maps.squeeze(1), (coils, size, size), coil_batch=0)
    A = linop.Sampling(pattern.squeeze(), S.oshape) @ S
    return kspace, maps, A, (y * (1.0 / scale)).squeeze(1), scale


def _wavelet(**kwargs):
    return prox.Wavelet(axes=(-1, -2), weight=0.01, **kwargs)


def _tv(weight=0.01):
    return prox.TotalVariation(axes=(-1, -2), weight=weight)


def _llr():
    return prox.LocallyLowRank(axes=(-1, -2), weight=0.01, block=4)


#: One configuration of ``pics``, as the tool's arguments and as a solver built
#: from the data scaling (which only PRIDU reads).  The same terms go to both.
_CONFIGURATIONS = [
    ("plain", {}, lambda scale: optim.CG(maxiter=20)),
    ("tikhonov", {"l2": 0.1}, lambda scale: optim.CG(0.1, maxiter=20)),
    (
        "wavelet admm",
        {"regularizers": _wavelet(), "solver": "admm"},
        lambda scale: optim.ADMM(_wavelet(), maxiter=20),
    ),
    (
        "wavelet fista",
        {"regularizers": _wavelet(), "solver": "fista"},
        lambda scale: optim.FISTA(_wavelet(), maxiter=20),
    ),
    (
        "wavelet ist",
        {"regularizers": _wavelet(), "solver": "ist"},
        lambda scale: optim.IST(_wavelet(), maxiter=20),
    ),
    (
        "no cycle spinning",
        {"regularizers": _wavelet(randshift=False), "solver": "fista"},
        lambda scale: optim.FISTA(_wavelet(randshift=False), maxiter=20),
    ),
    (
        "tv pridu",
        {"regularizers": _tv(), "solver": "pridu"},
        lambda scale: optim.PRIDU(_tv(), maxiter=20, sigma_tau_ratio=scale),
    ),
    (
        "locally low rank",
        {"regularizers": _llr(), "solver": "admm"},
        lambda scale: optim.ADMM(_llr(), maxiter=20),
    ),
    (
        "two terms",
        {"regularizers": [_wavelet(), _tv(0.005)], "solver": "admm"},
        lambda scale: optim.ADMM([_wavelet(), _tv(0.005)], maxiter=20),
    ),
]


@pytest.mark.parametrize(
    "theirs,ours", [(a, b) for _, a, b in _CONFIGURATIONS], ids=[n for n, _, _ in _CONFIGURATIONS]
)
def test_an_assembled_pics_is_the_tool_to_the_last_bit(theirs, ours, _whole_coil_operator):
    """Not close: the same bits.  A difference in the last place would mean
    some step was done twice, once by BART and once by this package."""
    kspace, maps, A, y, scale = _pics_problem()
    tool = bt.pics(kspace, maps, maxiter=20, **theirs).squeeze()
    assembled = ours(scale)(y, A).squeeze()
    assert torch.equal(assembled, tool), (
        f"maximum difference {float((assembled - tool).abs().max()):.3e}"
    )


def test_the_estimated_scaling_is_the_one_the_tool_estimates():
    kspace = bt.phantom(24, coils=4, kspace=True)
    pattern = bt.pattern(kspace)
    y = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
    # What the tool prints at debug level 1 for this data.
    assert optim.data_scaling(y) == pytest.approx(5490.628906, rel=1e-6)


def test_the_scaling_for_a_trajectory_is_the_other_branch():
    """``pics`` reads a non-Cartesian scaling off ``A^H y``; BART has no tool
    for it, so it is reached through the library."""
    n = 16
    traj = bt.traj(readout=n, spokes=24, radial=True)
    maps = _rand(2, n, n)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    A = linop.Sense(maps, (2, n, n), traj=traj)
    y = A(_rand(1, n, n))
    scale = optim.data_scaling(y, A=A)
    assert scale > 0
    assert optim.data_scaling(y * 4.0, A=A) == pytest.approx(4 * scale, rel=1e-5)


def test_a_wavelet_term_reused_answers_as_a_freshly_built_one():
    """BART's wavelet threshold draws its shifts from a generator of its own;
    the solve rewinds it, so a kept term answers as a fresh one would."""
    A = _unitary()
    y = _rand(8, 8)
    fista = optim.FISTA(_wavelet(), maxiter=20, step=1.0)
    assert torch.equal(fista(y, A), fista(y, A))


def test_cycle_spinning_is_on_as_it_is_for_the_tool():
    """``pics -n`` turns it off there, and it changes the answer."""
    A = _unitary()
    y = _rand(8, 8)
    spun = optim.FISTA(prox.Wavelet((-1, -2), 0.05), maxiter=20, step=1.0)(y, A)
    still = optim.FISTA(prox.Wavelet((-1, -2), 0.05, randshift=False), maxiter=20, step=1.0)(y, A)
    assert not torch.equal(spun, still)


def test_pridu_is_given_the_scaling_the_data_was_divided_by(_whole_coil_operator):
    """It balances its two steps with it, so it changes the iteration."""
    kspace, maps, A, y, scale = _pics_problem()
    term = prox.TotalVariation(axes=(-1, -2), weight=0.01)
    assert not torch.equal(
        optim.PRIDU(term, maxiter=20, sigma_tau_ratio=scale)(y, A),
        optim.PRIDU(term, maxiter=20)(y, A),
    )
