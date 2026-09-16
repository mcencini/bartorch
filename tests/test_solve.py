"""The solvers in ``bartorch.optim``, which are BART's and not this package's.

``bartorch_solve`` hands BART's ``lsqr2`` the encoding, the proximal operators
and the iteration ``italgo_config`` makes, which is what ``pics`` calls.

Two things are checked.  That the mechanism is BART's: the loop does not cross
back into Python, every iteration is reachable, a term means what its ``-R``
string means, and nothing is rebuilt per solve.  And that a reconstruction
assembled from operators, terms and a solver is ``bart pics`` to the bit,
across fifteen configurations (``test_an_assembled_pics_is_the_tool_to_the_last_bit``).
"""

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _dispatch, linop, optim, priors
from bartorch.linop import basic


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _unitary():
    """An operator whose normal is the identity, so the answer is known."""
    return linop.FFT((8, 8), axes=(-1, -2))


#: The coefficient axis of a subspace image as ``pics`` holds it:
#: ``COEFF_DIM`` is BART's sixth, so it is the seventh from the end.
_BART_COEFFS = -7

#: The coefficient axis of the subspace image an operator here takes,
#: ``(coeffs, z, y, x)``: the one before the three spatial axes.
_COEFFS = -4


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

    P = linop.LinearOperator.from_callbacks((8, 8), (8, 8), forward, adjoint)
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
    A = linop.NoncartesianSense(maps, (n, n), traj=traj)
    y = A(_rand(n, n))
    term = priors.Wavelet(axes=(-1, -2), weight=0.01)
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


def test_the_iteration_count_comes_back_from_the_library():
    """``steps`` is how many iterations ``conjgrad`` ran, counted in C.

    It cannot be worked out from outside: ``conjgrad`` stops on its own
    tolerance, and the alternating-direction solver budgets by the total
    across a whole run.  So the solve counts them with a monitor of its own
    and hands the number back.
    """
    diag = torch.linspace(0.1, 1.0, 64).to(torch.complex64).reshape(8, 8)
    A = linop.Diagonal(diag, (8, 8))
    y = A(_rand(8, 8))

    spent: list[int] = []
    optim.CG(maxiter=5)(y, A, steps=spent)
    assert spent == [5]

    # An encoding whose normal is the identity is solved long before the
    # budget runs out, and the count says so rather than repeating it back.
    easy: list[int] = []
    optim.CG(maxiter=20)(_rand(8, 8), _unitary(), steps=easy)
    assert 0 < easy[0] < 20

    # And a tolerance stops it sooner still.
    loose: list[int] = []
    optim.CG(maxiter=20, tol=0.5)(y, A, steps=loose)
    assert loose[0] < spent[0]


def test_asking_for_the_count_does_not_change_the_answer():
    diag = torch.linspace(0.1, 1.0, 64).to(torch.complex64).reshape(8, 8)
    A = linop.Diagonal(diag, (8, 8))
    y = A(_rand(8, 8))
    counted: list[int] = []
    assert torch.equal(optim.CG(maxiter=7)(y, A, steps=counted), optim.CG(maxiter=7)(y, A))
    assert counted


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
    got = make(priors.Wavelet(axes=(-1, -2), weight=0.01))(y, A)
    assert got.shape == A.ishape
    assert torch.isfinite(got.abs()).all()


def test_a_heavier_weight_shrinks_the_answer():
    A = _unitary()
    y = _rand(8, 8)
    # This operator's normal is the identity, so one is the right step.
    light = optim.FISTA(priors.Wavelet(axes=(-1, -2), weight=0.001), maxiter=30, step=1.0)(y, A)
    heavy = optim.FISTA(priors.Wavelet(axes=(-1, -2), weight=0.5), maxiter=30, step=1.0)(y, A)
    assert heavy.abs().sum() < light.abs().sum()


def test_several_terms_are_taken_together():
    terms = [
        priors.Wavelet(axes=(-1, -2), weight=0.01),
        priors.TotalVariation(axes=(-1, -2), weight=0.01),
    ]
    got = optim.ADMM(terms, maxiter=5)(_rand(8, 8), _unitary())
    assert torch.isfinite(got.abs()).all()


@pytest.mark.parametrize(
    "term",
    [
        priors.Wavelet(axes=(-1, -2), weight=0.01),
        priors.TotalVariation(axes=(-1, -2), weight=0.01),
        priors.LocallyLowRank(axes=(-1, -2), weight=0.01),
        priors.Laplace(axes=(-1, -2), weight=0.01),
        priors.L1(weight=0.01),
        priors.L2(weight=0.1),
        priors.NonNegative(),
        priors.ImaginaryL1(weight=0.01),
    ],
    ids=lambda t: type(t).__name__,
)
def test_every_term_bart_has_can_be_asked_for(term):
    got = optim.ADMM(term, maxiter=5)(_rand(8, 8), _unitary())
    assert torch.isfinite(got.abs()).all()


def test_a_term_is_built_once_and_handed_over_as_it_stands():
    """The object owns what BART made of it, so a second solve builds nothing."""
    term = priors.Wavelet(axes=(-1, -2), weight=0.01)
    first = term.build((8, 8))
    assert term.build((8, 8)) == first

    A = _unitary()
    y = _rand(8, 8)
    fista = optim.FISTA(term, maxiter=20, step=1.0)
    once = fista(y, A)
    twice = fista(y, A)
    assert torch.equal(once, twice)
    assert term.build((8, 8)) == first


def _extending():
    return [
        priors.TotalGeneralizedVariation((-1, -2), 0.01),
        priors.InfimalConvolutionTV((-1, -2, -7), 0.01),
        priors.InfimalConvolutionTGV((-1, -2, -7), 0.01),
    ]


@pytest.mark.parametrize("term", _extending(), ids=repr)
@pytest.mark.parametrize("solver", [optim.ADMM, optim.PRIDU])
def test_a_term_that_adds_unknowns_goes_to_the_iterations_given_a_transform(term, solver):
    """Total generalized variation and the infimal convolutions extend the
    optimization variable, so they are several penalties over a vector larger
    than the image.  ``iter2_admm`` and ``iter2_chambolle_pock`` are the two
    iterations BART hands a term's transform to, and they take them."""
    assert solver(term).regularizers == [term]


@pytest.mark.parametrize("term", _extending(), ids=repr)
@pytest.mark.parametrize("solver", [optim.IST, optim.FISTA, optim.NIHT])
def test_no_other_iteration_takes_a_term_that_adds_unknowns(term, solver):
    """``italgo_choose`` sends these to the alternating directions for the same
    reason: nothing else is given the transform that reaches the extra
    unknowns."""
    with pytest.raises(TypeError, match="auxiliary variables"):
        solver(term)


@pytest.mark.parametrize(
    "term",
    [priors.InfimalConvolutionTV((-1, -2), 0.01), priors.InfimalConvolutionTGV((-1, -2), 0.01)],
    ids=repr,
)
def test_an_infimal_convolution_needs_an_axis_of_each_kind(term):
    """``ictv_reg`` and ``ictgv_reg`` open with ``assert(0 != (flags &
    FFT_FLAGS))`` and ``assert(0 != (flags & ~FFT_FLAGS))`` -- the convolution
    separates what is smooth over the spatial axes from what is smooth over the
    others, so it needs both.  An assertion ends the process, so the same
    question is asked before BART is reached."""
    A = linop.FFT((1, 8, 8), axes=(-1, -2))
    with pytest.raises(ValueError, match="at least one of the image's last three axes"):
        optim.ADMM(term, maxiter=4)(_rand(1, 8, 8), A)


@pytest.mark.parametrize(
    "term,other,flag",
    [
        (
            lambda coeffs: priors.TotalGeneralizedVariation((-1, -2), 0.01, alpha=(2.0, 0.5)),
            lambda coeffs: priors.TotalGeneralizedVariation((-1, -2), 0.01),
            {"alpha": (2.0, 0.5)},
        ),
        (
            lambda coeffs: priors.InfimalConvolutionTV((-1, -2, coeffs), 0.01, gamma=(0.3, 2.0)),
            lambda coeffs: priors.InfimalConvolutionTV((-1, -2, coeffs), 0.01),
            {"gamma": (0.3, 2.0)},
        ),
    ],
    ids=["alpha", "gamma"],
)
def test_the_pairs_pics_takes_once_reach_the_solve(term, other, flag):
    """``--alpha`` and ``--gamma`` live on ``struct opt_reg_s`` and not on a
    term, so they travel beside the set rather than inside it.  Against the
    tool given the same flag, and against the default, which they change.

    Each term is made for the layout it is handed: the coefficient axis is
    ``_BART_COEFFS`` for the tool and ``_COEFFS`` for the operator."""
    kspace, maps, basis, A, y = _subspace_problem()
    tool = bt.pics(
        kspace, maps, basis=basis, maxiter=20, regularizers=term(_BART_COEFFS), solver="admm"
    )
    assembled = optim.ADMM(term(_COEFFS), maxiter=20)(y, A)
    assert torch.equal(assembled, tool.reshape(assembled.shape)), (
        f"maximum difference {float((assembled - tool.reshape(assembled.shape)).abs().max()):.3e}"
    )
    assert not torch.equal(assembled, optim.ADMM(other(_COEFFS), maxiter=20)(y, A))


def test_two_terms_cannot_ask_for_different_pairs():
    """One ``--alpha`` for the whole set, as for the block size."""
    A = linop.FFT((1, 8, 8), axes=(-1, -2))
    solver = optim.ADMM(
        [
            priors.TotalGeneralizedVariation((-1, -2), 0.01, alpha=(2.0, 0.5)),
            priors.TotalGeneralizedVariation(0, 0.01),
        ],
        maxiter=4,
    )
    with pytest.raises(ValueError, match="alpha is one pair"):
        solver(_rand(1, 8, 8), A)


def test_two_terms_cannot_ask_for_different_shared_options():
    """`opt_reg_configure` takes one block size, one wavelet family and one
    shift mode for the whole set -- ``pics`` has a single ``-b`` and a single
    ``-w`` -- so two terms disagreeing is refused rather than one of them
    silently winning."""
    A = linop.FFT((1, 8, 8), axes=(-1, -2))
    solver = optim.ADMM(
        [
            priors.TotalGeneralizedVariation((-1, -2), 0.01),
            priors.Wavelet((-1, -2), 0.01, family="haar"),
            priors.LocallyLowRank((-1, -2), 0.01, block=4),
        ],
        maxiter=4,
    )
    with pytest.raises(ValueError, match="one block size"):
        solver(_rand(1, 8, 8), A)


def test_a_term_that_adds_unknowns_refuses_a_tracked_right_hand_side():
    """Its penalties are BART's proximal operators, which have no backward pass;
    frozen, the gradient is the one with them held fixed."""
    A = linop.FFT((1, 8, 8), axes=(-1, -2))
    y = _rand(1, 8, 8).requires_grad_()
    term = priors.TotalGeneralizedVariation((-1, -2), 0.01)
    with pytest.raises(ValueError, match="no backward pass"):
        optim.ADMM(term, maxiter=4)(y, A)
    made = optim.ADMM(priors.frozen(term), maxiter=4)(y, A)
    made.abs().square().sum().backward()
    assert torch.isfinite(y.grad).all() and 0 < y.grad.abs().sum()


@pytest.mark.parametrize(
    "solver",
    [
        lambda t: optim.ADMM(t, maxiter=20),
        lambda t: optim.PRIDU(t, maxiter=12),
        lambda t: optim.PRIDU(t, maxiter=12, adaptive_step=True),
    ],
    ids=["admm", "pridu", "pridu adaptive"],
)
@pytest.mark.parametrize(
    "term",
    [
        lambda coeffs: priors.TotalGeneralizedVariation((-1, -2), 0.01),
        lambda coeffs: priors.InfimalConvolutionTV((-1, -2, coeffs), 0.01),
        lambda coeffs: priors.InfimalConvolutionTGV((-1, -2, coeffs), 0.01),
        lambda coeffs: [
            priors.Wavelet((-1, -2), 0.01),
            priors.TotalGeneralizedVariation((-1, -2), 0.01),
        ],
    ],
    ids=["tgv", "ictv", "ictgv", "wavelet and tgv"],
)
def test_a_block_walks_the_unknowns_a_term_adds_as_the_library_does(solver, term):
    """The image followed by the unknowns, in one vector, with the encoding
    chained onto an extract of its front, as ``pics.c`` chains it."""
    _, _, _, A, y = _subspace_problem()
    configured = solver(term(_COEFFS))
    assert torch.equal(configured(y, A), configured._in_library(y, A))


def test_a_batch_walks_each_items_unknowns():
    A = linop.FFT((1, 8, 8), axes=(-1, -2))
    data = torch.stack([A(_rand(1, 8, 8)) for _ in range(2)])
    block = optim.ADMMBlock(priors.TotalGeneralizedVariation((-1, -2), 0.01), cg_maxiter=4)

    def run(y):
        state = block.start(y, A)
        for _ in range(3):
            state = block(state, A)
        return block.output(state, A)

    batched = run(data)
    assert batched.shape == (2, 1, 8, 8)
    for i in range(2):
        assert torch.equal(batched[i], run(data[i]))


def test_a_term_that_adds_unknowns_takes_no_preconditioner():
    """The preconditioner maps the image, and the step walks a longer vector."""
    A = linop.FFT((1, 8, 8), axes=(-1, -2))
    solver = optim.ADMM(
        priors.TotalGeneralizedVariation((-1, -2), 0.01),
        maxiter=4,
        precond=linop.Identity((1, 8, 8)),
    )
    with pytest.raises(ValueError, match="no preconditioner"):
        solver(_rand(1, 8, 8), A)
    with pytest.raises(ValueError, match="no preconditioner"):
        solver._in_library(_rand(1, 8, 8), A)


@pytest.mark.parametrize(
    "term,ndim,string",
    [
        (priors.Wavelet(axes=(-1, -2), weight=0.01), 2, "W:3:0:0.01"),
        (priors.Wavelet(axes=(-1, -2), weight=0.01, joint_axes=0), 3, "W:3:4:0.01"),
        (priors.Wavelet(axes=0, weight=0.01), 3, "W:4:0:0.01"),
        (priors.L1(0.02, joint_axes=-3), 3, "I:4:0.02"),
        (priors.L2(0.5), 2, "Q:0.5"),
        (priors.NonNegative(), 2, "S"),
        (priors.WaveletNIHT(axes=(-1, -2), count=10), 2, "H:3:0:10"),
        (priors.TotalGeneralizedVariation((-1, -2), 0.01), 2, "G:3:0:0.01"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_term_is_the_string_the_parser_reads(term, ndim, string):
    """``-R W:3:0:0.01`` is a letter, two bitmasks and a weight, and so is the
    object, with the axes written as axes."""
    assert term._argument(ndim) == string


def test_an_axis_is_an_axis_and_not_a_bitmask():
    term = priors.Wavelet(axes=(-1, -2), weight=0.01)
    assert term._flags(ndim=2) == (3, 0)
    assert term._flags(ndim=3) == (3, 0)
    assert priors.Wavelet(axes=0, weight=0.01)._flags(ndim=3) == (4, 0)


def test_a_string_says_what_to_use_instead():
    with pytest.raises(TypeError, match="priors.Wavelet"):
        optim.FISTA("W:3:0:0.01")


# --- what it refuses --------------------------------------------------------


def test_niht_cannot_run_because_bart_asserts_against_its_own_iteration():
    """`niht` applies the normal operator in place -- `iter_op_call(op, g, g)`
    at `iter/niht.c:85` and `:212` -- and the operator `lsqr2` hands it asserts
    `args[0] != args[1]` at `iter/lsqr.c:60`.  `bart pics -R H` fails the same
    way; the difference is that a tool runs under BART's error catcher and a
    solve here does not, so the assertion would end the process."""
    A = _unitary()
    with pytest.raises(NotImplementedError, match="iter/niht.c"):
        optim.NIHT(priors.ImageNIHT((-1, -2), count=8), maxiter=4)(_rand(8, 8), A)


def test_the_tool_reaches_the_same_assertion_and_survives_it():
    kspace = bt.phantom(16, coils=2, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    with pytest.raises(bartorch.BartError, match="lsqr.c"):
        bt.pics(kspace, maps, regularizers=priors.WaveletNIHT((-1, -2), count=20), maxiter=5)


def test_niht_takes_only_hard_thresholding_terms():
    with pytest.raises(TypeError, match="NIHT"):
        optim.NIHT(priors.Wavelet(axes=(-1, -2), weight=0.01))


def test_something_that_is_not_a_term_is_refused():
    with pytest.raises(TypeError, match="bartorch.priors"):
        optim.ADMM(object())


def test_a_wavelet_family_bart_does_not_have_is_refused_here():
    """``opt_reg_configure`` answers an unknown family with ``error()``."""
    with pytest.raises(ValueError, match="family"):
        priors.Wavelet(axes=(-1, -2), weight=0.01, family="db4")


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
        getattr(priors, name).kind
        for name in priors.__all__
        if isinstance(getattr(priors, name), type) and getattr(getattr(priors, name), "kind", "")
    }
    assert offered <= bart_knows


# --- against the tool -------------------------------------------------------


@pytest.fixture
def _whole_coil_operator():
    """BART's own SENSE operator, which is what the tool builds.

    The coil-slab operator gives the same answer by different arithmetic, so
    the bit-for-bit comparison is made against BART's.  Process-wide, hence
    put back afterwards.

    This is about arithmetic and not about convention: the samples come back
    modulated because ``modulated=True`` asks for that, whatever the slab.
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

    # BART's own convention and BART's own operator: the k-space above is
    # modulated as `pics` modulates it, and `coil_batch=0` is the arithmetic
    # the tool does rather than arithmetic that agrees with it.  The tool's
    # arrays are (coils, z, y, x) with one partition, so dropping z leaves the
    # operator's (coils, y, x) maps and k-space over a (y, x) image.
    S = linop.CartesianSense(maps.squeeze(1), (size, size), coil_batch=0, modulated=True)
    A = basic.Sampling(pattern.squeeze(), S.oshape) @ S
    return kspace, maps, A, (y * (1.0 / scale)).squeeze(1), scale


def _wavelet(**kwargs):
    return priors.Wavelet(axes=(-1, -2), weight=0.01, **kwargs)


def _tv(weight=0.01):
    return priors.TotalVariation(axes=(-1, -2), weight=weight)


def _llr():
    return priors.LocallyLowRank(axes=(-1, -2), weight=0.01, block=4)


def _tgv(weight=0.01):
    return priors.TotalGeneralizedVariation(axes=(-1, -2), weight=weight)


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
        "tv admm",
        {"regularizers": _tv(0.005), "solver": "admm"},
        lambda scale: optim.ADMM(_tv(0.005), maxiter=20),
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
    (
        "tgv admm",
        {"regularizers": _tgv(), "solver": "admm"},
        lambda scale: optim.ADMM(_tgv(), maxiter=20),
    ),
    (
        "tgv pridu",
        {"regularizers": _tgv(), "solver": "pridu"},
        lambda scale: optim.PRIDU(_tgv(), maxiter=20, sigma_tau_ratio=scale),
    ),
    # The path that adds unknowns builds the whole set inside the solve, from
    # the one block size, wavelet family and shift mode `opt_reg_configure`
    # takes -- so a term beside it that reads one of those has to reach it.
    (
        "tgv and a haar wavelet",
        {"regularizers": [_tgv(), _wavelet(family="haar")], "solver": "admm"},
        lambda scale: optim.ADMM([_tgv(), _wavelet(family="haar")], maxiter=20),
    ),
    (
        "tgv and no cycle spinning",
        {"regularizers": [_tgv(), _wavelet(randshift=False)], "solver": "admm"},
        lambda scale: optim.ADMM([_tgv(), _wavelet(randshift=False)], maxiter=20),
    ),
    (
        "tgv and low rank over blocks of four",
        {"regularizers": [_tgv(), _llr()], "solver": "admm"},
        lambda scale: optim.ADMM([_tgv(), _llr()], maxiter=20),
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


def _subspace_problem(size=16, coils=4, frames=4, coeffs=2):
    """A ``pics -B`` problem, and the same problem assembled from this package.

    The image is the coefficients of a temporal subspace, which sit on BART's
    ``COEFF_DIM`` for the tool and before the three spatial axes for the
    operator -- an axis that is not one of the three spatial ones.  The
    infimal convolutions need such an axis: they separate what is smooth over
    the spatial axes from what is smooth over the rest, and BART asserts that
    both exist.  So the operator keeps the tool's single partition as a z axis
    of one: its image is ``(coeffs, z, y, x)`` and its k-space
    ``(coils, frames, z, y, x)``, while the tool's arrays stay in BART's order.
    """
    base = bt.phantom(size, coils=coils, kspace=True)
    decay = torch.linspace(1.0, 0.5, frames).to(torch.complex64)
    kspace = base.reshape(1, 1, 1, coils, 1, size, size) * decay.reshape(1, frames, 1, 1, 1, 1, 1)

    mask = torch.zeros(1, frames, 1, 1, 1, size, 1, dtype=torch.complex64)
    for t in range(frames):
        mask[0, t, 0, 0, 0, t::2, 0] = 1.0
        mask[0, t, 0, 0, 0, size // 2 - 2 : size // 2 + 2, 0] = 1.0
    kspace = kspace * mask

    basis = torch.zeros(coeffs, frames, 1, 1, 1, 1, 1, dtype=torch.complex64)
    basis[0, :, 0, 0, 0, 0, 0] = 1.0
    basis[1, :, 0, 0, 0, 0, 0] = torch.linspace(-1, 1, frames)

    maps = bt.ecalib(base, maps=1)
    pattern = bt.pattern(kspace).reshape(1, frames, 1, 1, 1, size, size)
    y = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
    scale = optim.data_scaling(y)

    # `toeplitz=False` because the tool applies the encoding and its adjoint;
    # the closed-form normal is this package's shortcut and not BART's step.
    # A coil batch of one, because BART's whole-coil operator lays the samples
    # out with the frames before the coils and is refused with a frame axis.
    A = linop.CartesianSense(
        maps.reshape(coils, 1, size, size),
        (coeffs, 1, size, size),
        pattern=pattern.reshape(frames, 1, size, size),
        basis=basis.reshape(coeffs, frames),
        toeplitz=False,
        coil_batch=1,
        modulated=True,
    )
    # BART's (1, frames, 1, coils, 1, y, x) to the operator's (coils, frames, z, y, x).
    y = y.reshape(frames, coils, 1, size, size).transpose(0, 1).contiguous()
    return kspace, maps, basis, A, (y * (1.0 / scale)).reshape(A.oshape)


@pytest.mark.parametrize(
    "term",
    [
        lambda coeffs: priors.Wavelet((-1, -2), 0.01),
        lambda coeffs: priors.TotalGeneralizedVariation((-1, -2), 0.01),
        lambda coeffs: priors.InfimalConvolutionTV((-1, -2, coeffs), 0.01),
        lambda coeffs: priors.InfimalConvolutionTGV((-1, -2, coeffs), 0.01),
    ],
    ids=["wavelet", "tgv", "ictv", "ictgv"],
)
def test_an_extending_term_over_a_subspace_is_the_tool_to_the_last_bit(term):
    """The terms that add unknowns, against ``pics -B`` itself.  A wavelet term
    is there as the baseline: it adds nothing, and if it drifted the assembly
    would be what drifted, not the enlarged variable."""
    kspace, maps, basis, A, y = _subspace_problem()
    tool = bt.pics(
        kspace, maps, basis=basis, maxiter=20, regularizers=term(_BART_COEFFS), solver="admm"
    )
    assembled = optim.ADMM(term(_COEFFS), maxiter=20)(y, A)
    tool = tool.reshape(assembled.shape)
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
    A = linop.NoncartesianSense(maps, (n, n), traj=traj)
    y = A(_rand(n, n))
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
    spun = optim.FISTA(priors.Wavelet((-1, -2), 0.05), maxiter=20, step=1.0)(y, A)
    still = optim.FISTA(priors.Wavelet((-1, -2), 0.05, randshift=False), maxiter=20, step=1.0)(y, A)
    assert not torch.equal(spun, still)


def test_pridu_is_given_the_scaling_the_data_was_divided_by(_whole_coil_operator):
    """It balances its two steps with it, so it changes the iteration."""
    kspace, maps, A, y, scale = _pics_problem()
    term = priors.TotalVariation(axes=(-1, -2), weight=0.01)
    assert not torch.equal(
        optim.PRIDU(term, maxiter=20, sigma_tau_ratio=scale)(y, A),
        optim.PRIDU(term, maxiter=20)(y, A),
    )


# --- quadratic penalties on the conjugate-gradient solve ----------------------
#
# BART's conjugate gradients takes one weight and nothing else: `iter2_conjgrad`
# asserts it is handed no regularizing operators and no biases, and
# `lsqr2_create` builds `A^H A + lambda I`.  So a penalty with an operator or a
# bias is not passed to the iteration at all -- it is built into the operator
# the iteration is given, by stacking the terms under the encoding.
#
# What is checked is that the stack is the right problem, against the normal
# equations solved densely, and that stacking does not cost the encoding its
# own normal.


def _dense(op):
    """``op`` as a matrix, one column per basis vector of its domain."""
    import math

    size = math.prod(op.ishape)
    columns = []
    for i in range(size):
        e = torch.zeros(size, dtype=torch.complex64)
        e[i] = 1
        columns.append(op(e.reshape(op.ishape)).reshape(-1))
    return torch.stack(columns, dim=1)


def _normal_equations(A, y, terms, lambda_=0.0):
    """``(A^H A + lambda I + sum w G^H G)^-1 (A^H y + sum w G^H b)``, exactly."""
    M = _dense(A)
    lhs = M.conj().T @ M
    rhs = M.conj().T @ y.reshape(-1)
    if lambda_:
        lhs = lhs + lambda_ * torch.eye(lhs.shape[0], dtype=torch.complex64)
    for term in terms:
        G = term.operator if term.operator is not None else linop.Identity(A.ishape)
        Gm = _dense(G)
        lhs = lhs + term.weight * (Gm.conj().T @ Gm)
        if term.bias is not None:
            rhs = rhs + term.weight * (Gm.conj().T @ term.bias.reshape(-1))
    return torch.linalg.solve(lhs, rhs).reshape(A.ishape)


@pytest.fixture
def _small_encoding():
    torch.manual_seed(0)
    maps = _rand(3, 8, 8)
    A = linop.CartesianSense(maps, (8, 8))
    return A, _rand(*A.oshape)


def _quadratic_cases(A):
    torch.manual_seed(1)
    G = linop.FFT(A.ishape, axes=(-2, -1))
    prior = _rand(*A.ishape)
    return {
        "a weight alone": [optim.Tikhonov(0.3)],
        "a bias": [optim.Tikhonov(0.3, bias=prior)],
        "an operator": [optim.Tikhonov(0.3, operator=G)],
        "an operator and a bias": [optim.Tikhonov(0.3, operator=G, bias=_rand(*G.oshape))],
        "two terms": [optim.Tikhonov(0.3, bias=prior), optim.Tikhonov(0.05, operator=G)],
    }


@pytest.mark.parametrize("case", list(_quadratic_cases(linop.FFT((1, 8, 8), axes=(-2, -1)))))
def test_a_quadratic_penalty_is_the_normal_equations_it_claims(case, _small_encoding):
    A, y = _small_encoding
    terms = _quadratic_cases(A)[case]
    got = optim.CG(terms=terms, maxiter=300)(y, A)
    want = _normal_equations(A, y, terms)
    assert (got - want).abs().max() / want.abs().max() < 1e-4


def test_a_weight_and_a_term_go_together(_small_encoding):
    """``lambda_`` stays BART's own, and the terms are added to it."""
    A, y = _small_encoding
    terms = [optim.Tikhonov(0.2, operator=linop.FFT(A.ishape, axes=(-2, -1)))]
    got = optim.CG(0.05, terms=terms, maxiter=300)(y, A)
    want = _normal_equations(A, y, terms, lambda_=0.05)
    assert (got - want).abs().max() / want.abs().max() < 1e-4


def test_no_terms_is_the_solve_it_always_was(_small_encoding):
    """Nothing is stacked when there is nothing to stack."""
    A, y = _small_encoding
    torch.testing.assert_close(
        optim.CG(0.1, terms=None, maxiter=12)(y, A), optim.CG(0.1, maxiter=12)(y, A), rtol=0, atol=0
    )


def test_stacking_does_not_cost_the_encoding_its_own_normal():
    """The point of building the normal rather than letting the stack derive it.

    A Toeplitz encoding answers A^H A as a convolution with a point spread
    function, which is a different route from the transform and its adjoint --
    and a measurably different answer.  The stacked operator has to take the
    first route, or a Toeplitz encoding would quietly stop being one inside a
    regularized solve.
    """
    from bartorch.optim.linear import _stacked

    n, coils = 16, 4
    torch.manual_seed(0)
    maps = _rand(coils, n, n)
    traj = bt.traj(x=n, y=24, r=True)
    A = linop.NoncartesianSense(maps, (n, n), traj=traj, toeplitz=True)
    y = _rand(*A.oshape)
    G = linop.FFT(A.ishape, axes=(-2, -1))

    stacked, _ = _stacked(A, y, [optim.Tikhonov(0.3, operator=G)])
    x = _rand(*A.ishape)

    grams = A.gram()(x) + 0.3 * G.gram()(x)
    both = A.adjoint(A(x)) + 0.3 * G.adjoint(G(x))

    # The same operator, not merely the same answer.
    torch.testing.assert_close(stacked.normal(x), grams, rtol=0, atol=0)
    # And the two routes really do differ, so the check above has teeth.
    assert (grams - both).abs().max() / both.abs().max() > 1e-5


def test_the_stacked_data_is_the_terms_laid_end_to_end(_small_encoding):
    import math

    A, y = _small_encoding
    from bartorch.optim.linear import _stacked

    G = linop.FFT(A.ishape, axes=(-2, -1))
    bias = _rand(*G.oshape)
    stacked, data = _stacked(A, y, [optim.Tikhonov(0.25, operator=G, bias=bias)])

    assert stacked.oshape == (math.prod(A.oshape) + math.prod(G.oshape),)
    torch.testing.assert_close(data[: math.prod(A.oshape)], y.reshape(-1))
    torch.testing.assert_close(
        data[math.prod(A.oshape) :], math.sqrt(0.25) * bias.reshape(-1), rtol=1e-6, atol=1e-6
    )


def test_a_penalty_that_does_not_fit_the_encoding_is_refused(_small_encoding):
    A, y = _small_encoding
    with pytest.raises(ValueError, match="does not fit an encoding"):
        optim.CG(terms=optim.Tikhonov(0.1, operator=linop.FFT((1, 4, 4), axes=-1)))(y, A)


def test_a_negative_weight_is_not_a_penalty():
    with pytest.raises(ValueError, match="at least zero"):
        optim.Tikhonov(-1.0)


def test_cg_takes_quadratic_terms_and_not_proximal_ones(_small_encoding):
    A, y = _small_encoding
    with pytest.raises(TypeError, match="takes Tikhonov terms"):
        optim.CG(terms=priors.L1(0.1))(y, A)


def test_a_solver_says_what_it_was_given():
    assert "Tikhonov" in repr(optim.CG(terms=optim.Tikhonov(0.5)))
    assert "terms" not in repr(optim.CG(0.5))
