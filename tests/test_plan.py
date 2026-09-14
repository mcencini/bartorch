"""The encoding form, the plan it is lowered into, and which executor ran it.

A silent fallback passes every correctness test: an encoding that walks back
to BART's plain chain of operators answers with the same numbers and costs
several times as much.  So each case here checks both -- the result against
something outside BART, and the plan and the library's own counters against
the path it was meant to take.
"""

import subprocess
import sys

import pytest
import torch

import bartorch
from bartorch import _abi, _finufft, linop
from bartorch._lib import library
from bartorch.linop import plan as planner

COILS, Z, Y, X = 4, 3, 16, 12

requires_cuda = pytest.mark.skipif(
    not bartorch._cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


@pytest.fixture
def maps():
    torch.manual_seed(0)
    return _rand(COILS, Y, X)


@pytest.fixture
def pattern():
    torch.manual_seed(1)
    return (torch.rand(Y, 1) > 0.4).to(torch.complex64)


@pytest.fixture
def basis():
    torch.manual_seed(2)
    return _rand(3, 8)


def _counter(which):
    return library().bartorch_encoding_counter(which)


# --- the form the encodings are lowered into ---------------------------------


def test_every_encoding_reports_its_transform(maps, pattern):
    """The three built-in encodings, and the coil multiply with no transform."""
    traj = bartorch.tools.traj(x=X, y=8)
    psf = _rand(Y, 2 * X)
    cases = {
        "fft": linop.CartesianSense(maps, (Y, X), pattern=pattern),
        "nufft": linop.NoncartesianSense(maps, (Y, X), traj=traj),
        "wave": linop.WaveSense(maps, (Y, X), psf=psf, readout=2 * X),
        "none": linop.Coils(maps, (Y, X)),
    }
    for name, A in cases.items():
        assert A.plan.transform == name
        assert A.plan.fused, f"{name} fell back to BART's plain chain"


def test_the_factors_are_the_ones_the_encoding_was_given(maps, basis):
    frames = (torch.rand(8, Y, 1) > 0.4).to(torch.complex64)
    A = linop.CartesianSense(maps, (3, Y, X), pattern=frames, basis=basis)
    assert [f.name for f in A.plan.image] == ["sensitivities"]
    assert [f.name for f in A.plan.kspace] == ["basis", "pattern"]
    assert A.plan.contraction == "subspace" and A.plan.terms == 3
    assert A.plan.kspace[0].shape == (3, 8)


def test_kernels_are_streamed_rather_than_held(pattern):
    """A bank held as kernels is inflated a slab at a time, and the plan says so."""
    torch.manual_seed(3)
    kernels = _rand(COILS, 6, 6)
    A = linop.CartesianSense(kernels, (Y, X), pattern=pattern, kernels=True)
    bank = next(f for f in A.plan.image if f.name == "sensitivities")
    assert bank.held == "kernels"
    assert "coil kernels" in A.plan.streamed


def test_a_table_of_positions_is_a_compressed_factor(maps):
    torch.manual_seed(4)
    positions = torch.randint(0, Y, (5, 1))
    A = linop.CartesianSense(maps, (Y, X), positions=positions)
    table = next(f for f in A.plan.kspace if f.name == "positions")
    assert table.held == "table"
    assert A.plan.fused


# --- how the normal is applied -----------------------------------------------


def test_a_pattern_collapses_into_a_kernel(maps, pattern):
    assert linop.CartesianSense(maps, (Y, X), pattern=pattern).plan.normal == "kernel"


def test_without_toeplitz_the_normal_is_the_two_applications(maps, pattern):
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern, toeplitz=False)
    assert A.plan.normal == "applications"


def test_without_a_k_space_factor_the_normal_is_the_transforms_own(maps):
    assert linop.CartesianSense(maps, (Y, X)).plan.normal == "transform"


def test_a_wave_pattern_along_the_readout_has_no_closed_form(maps):
    """The kernel sits between the phase-encode transforms, so the readout cannot cancel."""
    torch.manual_seed(5)
    psf = _rand(Y, 2 * X)
    along = (torch.rand(Y, 2 * X) > 0.4).to(torch.complex64)
    across = (torch.rand(Y, 1) > 0.4).to(torch.complex64)
    A = linop.WaveSense(maps, (Y, X), psf=psf, readout=2 * X, pattern=along)
    B = linop.WaveSense(maps, (Y, X), psf=psf, readout=2 * X, pattern=across)
    assert (A.plan.normal, B.plan.normal) == ("applications", "kernel")


# --- which executor ran it ----------------------------------------------------


def test_the_slab_loop_is_what_ran_the_encoding(maps, pattern):
    """Counted in the library, not inferred from a timing."""
    library().bartorch_encoding_reset_counters()
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    assert (_counter(_abi.BARTORCH_ENCODING_BUILT), _counter(_abi.BARTORCH_ENCODING_CHAINED)) == (
        1,
        0,
    )

    x = _rand(Y, X)
    A(x)
    A.adjoint(A(x))
    A.normal(x)
    ran = (
        _counter(_abi.BARTORCH_ENCODING_FORWARD),
        _counter(_abi.BARTORCH_ENCODING_ADJOINT),
        _counter(_abi.BARTORCH_ENCODING_NORMAL),
    )
    assert ran == (2, 1, 1)


def test_a_form_the_loop_cannot_take_says_so(maps, pattern):
    """``coil_batch=0`` asks for BART's own operator over every coil at once."""
    library().bartorch_encoding_reset_counters()
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern, coil_batch=0)
    assert A.plan.executor == "chain" and not A.plan.fused
    assert (_counter(_abi.BARTORCH_ENCODING_BUILT), _counter(_abi.BARTORCH_ENCODING_CHAINED)) == (
        0,
        1,
    )


def test_a_coil_slab_is_what_the_plan_reports(maps, pattern):
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern, coil_batch=2)
    assert A.plan.coil_batch == 2 and "coils" in A.plan.streamed


# --- matching a composition ---------------------------------------------------


def test_a_diagonal_either_side_of_an_encoding_is_a_chain(maps, pattern):
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    w = _rand(1, Y, 1)
    c = _rand(Y, X)
    described = planner.describe(linop.Diagonal(w, A.oshape) @ A @ linop.Diagonal(c, A.ishape))
    assert described is not None and len(described.terms) == 1
    term = described.terms[0]
    assert term.encoding is A
    assert torch.equal(term.kspace, w) and torch.equal(term.image, c)


def test_a_sum_of_chains_is_a_sum_of_terms(maps, pattern):
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    terms = [linop.Diagonal(_rand(1, Y, 1), A.oshape) @ A for _ in range(3)]
    described = planner.describe(terms[0] + terms[1] + terms[2])
    assert described is not None and len(described.terms) == 3


def test_something_that_is_not_element_wise_has_no_description(maps, pattern):
    """A transform in front of the encoding is outside the form, and is left alone."""
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    assert planner.describe(A @ linop.FFT(A.ishape, axes=(-2, -1))) is None


def test_a_composition_the_planner_matches_is_one_encoding(maps, pattern):
    """The lowered operator is the sum written out, and it is fused."""
    torch.manual_seed(6)
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    b = _rand(3, 1, Y, X)
    c = _rand(3, Y, X)
    described = planner.Contract(A, b, c)

    fused = planner.lower(described)
    chained = planner.materialise(described)
    assert fused.plan.contraction == "segments" and fused.plan.fused
    assert chained.plan.contraction == "chained" and not chained.plan.fused

    x = _rand(Y, X)
    want = sum(b[term] * A(c[term] * x) for term in range(3))
    for built in (fused, chained):
        assert (built(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_fused_contraction_costs_one_application_and_a_chained_one_costs_a_term_each(
    maps, pattern
):
    """What composing has to cost nothing means, counted rather than timed."""
    torch.manual_seed(17)
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    described = planner.Contract(A, _rand(3, 1, Y, X), _rand(3, Y, X))
    fused, chained = planner.lower(described), planner.materialise(described)
    x = _rand(Y, X)

    library().bartorch_encoding_reset_counters()
    fused(x)
    assert _counter(_abi.BARTORCH_ENCODING_FORWARD) == 1

    library().bartorch_encoding_reset_counters()
    chained(x)
    assert _counter(_abi.BARTORCH_ENCODING_FORWARD) == 3


def test_terms_that_do_not_share_an_encoding_are_not_a_contraction(maps, pattern):
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    B = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    described = planner.Sum((planner.Chain(A, _rand(Y, X)), planner.Chain(B, _rand(Y, X))))
    assert planner.lower(described) is None


# --- the fused plan for the cases the design names ----------------------------


def test_a_grid_contraction_is_fused_and_a_coil_varying_one_is_not(maps, pattern):
    """Weights along the coils cannot be sliced with a slab, so the sum stands."""
    torch.manual_seed(7)
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    c = _rand(3, Y, X)
    fused = linop.FieldCorrected(A, coefficients=(_rand(3, 1, Y, X), c))
    stands = linop.FieldCorrected(A, coefficients=(_rand(3, COILS, Y, X), c))
    assert fused.plan.contraction == "segments" and fused.plan.fused
    assert stands.plan.contraction == "chained" and not stands.plan.fused


@pytest.mark.skipif(
    not _finufft.serves(), reason="a basis along the samples is a transform only FINUFFT computes"
)
def test_a_nufft_contraction_becomes_a_subspace_over_the_samples(maps):
    torch.manual_seed(8)
    traj = bartorch.tools.traj(x=X, y=8)
    E = linop.NoncartesianSense(maps, (X, X), traj=traj)
    b = _rand(3, *E.oshape[1:])
    c = _rand(3, X, X)
    A = linop.FieldCorrected(E, coefficients=(b, c))
    assert A.plan.transform == "nufft"
    assert A.plan.contraction == "subspace" and A.plan.terms == 3
    assert A.plan.normal == "kernel", "a kernel per pair of terms, not a transform each"
    assert A.plan.fused


def test_the_substitution_answers_before_anything_has_needed_one():
    """``serves()`` decides whether a test runs, so it must not answer "no" too early.

    The substitution installs itself on first use, so read in a process that
    has not built a NUFFT yet it would report that it does not serve -- and a
    test gated on it would skip where it should run, which is the silent
    fallback this file exists to catch.  It installs first.

    What is asserted is that the two agree, not that either is true: where the
    substitution declines for a reason of its own -- a macOS process, whose
    second OpenMP runtime it will not start -- the honest answer is that it
    does not serve, in a fresh process and in this one alike.
    """
    probe = (
        "from bartorch import _finufft\n"
        "from bartorch._lib import library\n"
        "asked = _finufft.serves()\n"  # nothing has needed a NUFFT yet
        "_finufft.install_once()\n"  # what the first one would have done
        "print(asked, bool(library().bartorch_finufft_usable_on(0)))\n"
    )
    fresh = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    asked, after = fresh.stdout.split()
    assert asked == after, fresh.stderr


def test_a_contraction_barts_gridder_cannot_serve_falls_back_to_the_sum(maps):
    """The planner lowers into no form the library cannot apply.

    BART's own gridder asserts that a basis is trivial over the sample axes,
    so a contraction over a NUFFT is the substitution's alone; without it the
    sum of chains answers, and the plan says the terms are chained.
    """
    torch.manual_seed(18)
    traj = bartorch.tools.traj(x=X, y=8)
    E = linop.NoncartesianSense(maps, (X, X), traj=traj)
    b, c = _rand(3, *E.oshape[1:]), _rand(3, X, X)

    x = _rand(X, X)
    want = sum(b[term] * E(c[term] * x) for term in range(3))
    with _finufft.barts_own_gridder():
        A = linop.FieldCorrected(E, coefficients=(b, c))
        assert A.plan.contraction == "chained" and not A.plan.fused
        assert (A(x) - want).abs().max() / want.abs().max() < 1e-5


@pytest.mark.parametrize(
    "case",
    [
        "cartesian",
        "cartesian subspace, dense pattern",
        "cartesian subspace, positions",
        "wave",
        "wave subspace, dense pattern",
        "wave subspace, positions",
        "non-cartesian subspace",
        "field-corrected non-cartesian",
        "field-corrected cartesian",
        "field-corrected wave",
    ],
)
def test_every_benchmarked_case_takes_the_fused_plan(case):
    """The cases the design's targets are measured on, at a size that fits a test."""
    # A contraction over a NUFFT puts its basis along the samples, which only
    # FINUFFT computes; every other case here has a route BART's gridder has.
    if case == "field-corrected non-cartesian" and not _finufft.serves():
        pytest.skip("a basis along the samples is a transform only FINUFFT computes")
    A = _benchmark_case(case)
    assert A.plan.fused, f"{case} did not take the fused plan: {A.plan}"


def _benchmark_case(case, device=None):
    """One of the design's benchmark cases, small.

    The same builds ``scripts/benchmark_encodings.py`` times, so a case that
    is measured is a case that is held to its plan here.  ``device`` is where
    the encoding is built; its arrays stay where they are either way.
    """
    torch.manual_seed(9)
    wx = 2 * X
    maps = _rand(COILS, Z, Y, X)
    shape = (Z, Y, X)
    basis = _rand(3, 8)
    dense = (torch.rand(Z, Y, 1) > 0.4).to(torch.complex64)
    frames = (torch.rand(8, Z, Y, 1) > 0.4).to(torch.complex64)
    positions = torch.stack([torch.randint(0, Z, (8, 5)), torch.randint(0, Y, (8, 5))], dim=-1)
    wave = _rand(Z, Y, wx)
    traj = bartorch.tools.traj(x=X, y=8)
    flat = _rand(COILS, Y, X)
    where = {"device": device}

    if case == "cartesian":
        return linop.CartesianSense(maps, shape, pattern=dense, **where)
    if case == "cartesian subspace, dense pattern":
        return linop.CartesianSense(maps, (3, *shape), pattern=frames, basis=basis, **where)
    if case == "cartesian subspace, positions":
        return linop.CartesianSense(maps, (3, *shape), positions=positions, basis=basis, **where)
    if case == "wave":
        return linop.WaveSense(maps, shape, psf=wave, readout=wx, pattern=dense, **where)
    if case == "wave subspace, dense pattern":
        pat = (torch.rand(8, Z, Y, 1) > 0.4).to(torch.complex64)
        return linop.WaveSense(
            maps, (3, *shape), psf=wave, readout=wx, pattern=pat, basis=basis, **where
        )
    if case == "wave subspace, positions":
        return linop.WaveSense(
            maps, (3, *shape), psf=wave, readout=wx, positions=positions, basis=basis, **where
        )
    if case == "non-cartesian subspace":
        many = traj.unsqueeze(0).expand(8, -1, -1, -1).contiguous()
        return linop.NoncartesianSense(flat, (3, Y, X), traj=many, basis=basis, **where)
    if case == "field-corrected non-cartesian":
        E = linop.NoncartesianSense(flat, (Y, X), traj=traj, **where)
        return linop.FieldCorrected(E, coefficients=(_rand(3, *E.oshape[1:]), _rand(3, Y, X)))
    if case == "field-corrected cartesian":
        mask = (torch.rand(Y, 1) > 0.4).to(torch.complex64)
        E = linop.CartesianSense(flat, (Y, X), pattern=mask, **where)
        return linop.FieldCorrected(E, coefficients=(_rand(3, 1, Y, X), _rand(3, Y, X)))
    if case == "field-corrected wave":
        E = linop.WaveSense(flat, (Y, X), psf=_rand(Y, wx), readout=wx, **where)
        return linop.FieldCorrected(E, coefficients=(_rand(3, 1, Y, wx), _rand(3, Y, X)))
    raise AssertionError(case)


# --- the numbers, against something outside BART ------------------------------


def test_a_fused_cartesian_encoding_is_the_written_out_model(maps, pattern):
    """``y[c] = P . F(S[c] x)``, with torch's own transform."""
    torch.manual_seed(10)
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    assert A.plan.fused
    x = _rand(Y, X)

    coil = maps * x
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(coil, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    want = pattern * spectrum
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_fused_normal_is_the_kernel_written_out(maps, pattern):
    """``(A^H A x)`` with the pattern collapsed once, against the sum over the samples."""
    torch.manual_seed(11)
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    assert A.plan.normal == "kernel"
    x = _rand(Y, X)

    coil = maps * x
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(coil, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    kept = (pattern.abs() ** 2) * spectrum
    back = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kept, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    want = (maps.conj() * back).sum(0)
    assert (A.normal(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_fused_contraction_is_the_sum_it_stands_for(maps, pattern):
    """The fused encoding against the same sum applied term by term in torch."""
    torch.manual_seed(12)
    E = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    b, c = _rand(3, 1, Y, X), _rand(3, Y, X)
    A = linop.FieldCorrected(E, coefficients=(b, c))
    assert A.plan.fused

    x = _rand(Y, X)
    coils = maps[None] * (c * x)[:, None]
    spectra = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(coils, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    want = (b * pattern * spectra).sum(0)
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-5


def test_the_fused_and_the_chained_contraction_agree(maps, pattern):
    """The fallback is the same operator, which is what makes it a fallback."""
    torch.manual_seed(13)
    E = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    b, c = _rand(3, 1, Y, X), _rand(3, Y, X)
    described = planner.Contract(E, b, c)
    fused, chained = planner.lower(described), planner.materialise(described)

    x, y = _rand(Y, X), _rand(*E.oshape)
    for one, other in ((fused(x), chained(x)), (fused.adjoint(y), chained.adjoint(y))):
        assert (one - other).abs().max() / other.abs().max() < 1e-5


def test_a_fused_wave_encoding_is_the_chain_written_out(maps):
    """Zero-fill, readout transform, point spread function, phase-encode transform."""
    torch.manual_seed(14)
    wx = 2 * X
    psf = _rand(Y, wx)
    A = linop.WaveSense(maps, (Y, X), psf=psf, readout=wx, centred=True)
    assert A.plan.fused and A.plan.transform == "wave"

    x = _rand(Y, X)
    coil = maps * x
    wide = torch.zeros(COILS, Y, wx, dtype=torch.complex64)
    start = wx // 2 - X // 2
    wide[..., start : start + X] = coil
    along = torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(wide, dim=-1), dim=-1, norm="ortho"), dim=-1
    )
    spread = psf * along
    want = torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(spread, dim=-2), dim=-2, norm="ortho"), dim=-2
    )
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_fused_subspace_encoding_is_the_frames_written_out(maps, basis, pattern):
    """``y[c, t] = P . sum_a phi[a, t] F(S[c] x[a])``."""
    torch.manual_seed(15)
    frames = (torch.rand(8, Y, 1) > 0.4).to(torch.complex64)
    A = linop.CartesianSense(maps, (3, Y, X), pattern=frames, basis=basis)
    assert A.plan.fused and A.plan.contraction == "subspace"

    x = _rand(3, Y, X)
    coils = maps[:, None] * x[None]
    spectra = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(coils, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    contracted = torch.einsum("at,carx->ctrx", basis, spectra)
    want = frames[None] * contracted
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-5


def test_the_algebra_carries_the_plan_through(maps, pattern):
    """An adjoint, a normal and a chain are the same encoding, so they say so."""
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    for built in (A.H, A.gram(), A @ linop.Identity(A.ishape), 2.0 * A):
        assert built.plan == A.plan


def test_two_encodings_in_one_composition_have_no_single_plan(maps, pattern):
    """The report is one encoding's; a composition of two is not one of them."""
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    B = linop.CartesianSense(maps, (Y, X), pattern=pattern, coil_batch=0)
    assert (A.H @ B).plan is None


def test_a_plan_reads_as_a_sentence(maps, pattern):
    """It is what a caller is shown, so it says the parts rather than the fields."""
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    text = repr(A.plan)
    for part in ("transform=fft", "image=sensitivities", "kspace=pattern", "executor=slab"):
        assert part in text


# --- on a card ----------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "case",
    [
        "cartesian",
        "cartesian subspace, dense pattern",
        "cartesian subspace, positions",
        "wave",
        "wave subspace, dense pattern",
        "wave subspace, positions",
        "non-cartesian subspace",
        "field-corrected non-cartesian",
        "field-corrected cartesian",
        "field-corrected wave",
    ],
)
def test_on_a_card_every_benchmarked_case_takes_the_same_plan(case):
    """The plan is the form's, not the device's: a card changes where, not what."""
    host = _benchmark_case(case)
    device = _benchmark_case(case, device="cuda")
    assert device.plan.fused
    assert (device.plan.transform, device.plan.contraction, device.plan.normal) == (
        host.plan.transform,
        host.plan.contraction,
        host.plan.normal,
    )


@requires_cuda
def test_on_a_card_a_fused_encoding_is_the_host_one(maps, pattern):
    torch.manual_seed(16)
    host = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    card = linop.CartesianSense(maps, (Y, X), pattern=pattern, device="cuda")
    assert card.plan.fused

    x = _rand(Y, X)
    for one, other in ((card(x), host(x)), (card.normal(x), host.normal(x))):
        assert (one - other).abs().max() / other.abs().max() < 1e-4


@requires_cuda
def test_on_a_card_the_slab_executor_is_still_what_ran_it(maps, pattern):
    library().bartorch_encoding_reset_counters()
    A = linop.CartesianSense(maps, (Y, X), pattern=pattern, device="cuda")
    A.normal(_rand(Y, X))
    assert _counter(_abi.BARTORCH_ENCODING_BUILT) == 1
    assert _counter(_abi.BARTORCH_ENCODING_CHAINED) == 0
    assert _counter(_abi.BARTORCH_ENCODING_NORMAL) == 1


# --- a composition written by hand, rather than by a constructor --------------


def _terms(E, b, c):
    """``sum_l diag(b_l) E diag(c_l)`` written with ``@`` and ``+``."""
    out = None
    for term in range(int(b.shape[0])):
        built = linop.Diagonal(b[term], E.oshape) @ E @ linop.Diagonal(c[term], E.ishape)
        out = built if out is None else out + built
    return out


def test_composing_builds_a_description_and_nothing_else(maps, pattern):
    """``@`` and ``+`` work the shapes out; nothing is built until something needs it."""
    E = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    built = linop.Diagonal(_rand(Y, X), E.ishape)
    composed = E @ built

    assert "_h" not in composed.__dict__
    assert composed.ishape == E.ishape and composed.oshape == E.oshape
    composed(_rand(Y, X))
    assert "_h" in composed.__dict__


def test_a_sum_written_by_hand_is_the_contraction_the_planner_lowers(maps, pattern):
    """The composition a caller writes and the description a fit hands over are one operator."""
    torch.manual_seed(23)
    E = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    b, c = _rand(3, 1, Y, X), _rand(3, Y, X)

    composed = _terms(E, b, c)
    assert composed.plan.contraction == "segments"
    assert composed.plan.terms == 3
    assert composed.plan.fused

    fused = planner.lower(planner.Contract(E, b, c))
    x = _rand(Y, X)
    assert torch.equal(composed(x), fused(x))


def test_a_sum_written_by_hand_is_the_sum_written_out(maps, pattern):
    """Against the model itself: the segments summed with torch's own transform."""
    torch.manual_seed(24)
    E = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    b, c = _rand(3, 1, Y, X), _rand(3, Y, X)
    x = _rand(Y, X)

    want = torch.zeros(COILS, Y, X, dtype=torch.complex64)
    for term in range(3):
        coils = maps * (c[term] * x)
        shifted = torch.fft.ifftshift(coils, dim=(-2, -1))
        k = torch.fft.fftshift(torch.fft.fft2(shifted, norm="ortho"), dim=(-2, -1))
        want = want + b[term] * pattern * k

    got = _terms(E, b, c)(x)
    assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_a_composition_costs_one_application_however_many_terms(maps, pattern):
    """The counted form of composing costing nothing, for a sum a caller wrote."""
    torch.manual_seed(25)
    E = linop.CartesianSense(maps, (Y, X), pattern=pattern)
    composed = _terms(E, _rand(4, 1, Y, X), _rand(4, Y, X))

    library().bartorch_encoding_reset_counters()
    composed(_rand(Y, X))
    assert _counter(_abi.BARTORCH_ENCODING_FORWARD) == 1


# --- the compositions the design names ----------------------------------------


def _fft2(x):
    shifted = torch.fft.ifftshift(x, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fft2(shifted, norm="ortho"), dim=(-2, -1))


def test_a_shot_phase_is_a_contraction_over_the_shots(maps):
    """Multishot: one image, a phase per shot, and the samples each shot took.

    Against the shots written out, with torch's own transform.
    """
    torch.manual_seed(30)
    shots = 3
    E = linop.CartesianSense(maps, (Y, X), pattern=torch.ones(Y, 1, dtype=torch.complex64))

    phase = _rand(shots, Y, X)
    taken = torch.zeros(shots, 1, Y, 1, dtype=torch.complex64)
    for shot in range(shots):
        taken[shot, 0, shot::shots, 0] = 1

    A = _terms(E, taken, phase)
    assert A.plan.contraction == "segments" and A.plan.terms == shots
    assert A.plan.fused

    x = _rand(Y, X)
    want = torch.zeros(COILS, Y, X, dtype=torch.complex64)
    for shot in range(shots):
        want = want + taken[shot] * _fft2(maps * (phase[shot] * x))
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-5


def test_an_echo_phase_with_a_basis_is_a_contraction_over_the_frames(maps, basis):
    """A per-voxel phase per frame cannot move to the k-space side, so it is a term each.

    The basis still contracts the coefficients inside each term, which is what
    makes this one encoding per frame rather than one per frame and
    coefficient.
    """
    torch.manual_seed(31)
    coeffs, frames = int(basis.shape[0]), int(basis.shape[1])
    pattern = torch.ones(frames, Y, 1, dtype=torch.complex64)
    E = linop.CartesianSense(maps, (coeffs, Y, X), pattern=pattern, basis=basis)

    phase = _rand(frames, 1, Y, X)
    selector = torch.zeros(frames, 1, frames, 1, 1, dtype=torch.complex64)
    for frame in range(frames):
        selector[frame, 0, frame] = 1

    A = _terms(E, selector, phase)
    assert A.plan.contraction == "segments" and A.plan.terms == frames
    assert A.plan.fused

    x = _rand(coeffs, Y, X)
    want = torch.zeros(COILS, frames, Y, X, dtype=torch.complex64)
    for frame in range(frames):
        for coeff in range(coeffs):
            want[:, frame] += basis[coeff, frame] * _fft2(maps * (phase[frame, 0] * x[coeff]))
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-5


def test_a_weight_that_differs_between_sets_is_left_to_the_sum(pattern):
    """The sets are summed before the image factor, so a factor along them is not this form.

    What it must not be is fused and wrong: the slab contracts the sets away
    with ``md_ztenmul2`` before the contraction's image factor is reached, so
    the answer has to come from the sum of the terms instead.
    """
    torch.manual_seed(32)
    sets = 2
    sensitivities = _rand(sets, COILS, Y, X)
    E = linop.CartesianSense(sensitivities, (sets, Y, X), pattern=pattern)
    assert E.sets == sets

    picked = torch.zeros(sets, sets, 1, 1, dtype=torch.complex64)
    for s in range(sets):
        picked[s, s] = 1
    phase = _rand(sets, 1, Y, 1)

    A = _terms(E, phase, picked)
    assert A.plan.contraction == "chained" and A.plan.terms == sets
    assert not A.plan.fused

    x = _rand(sets, Y, X)
    want = torch.zeros(COILS, Y, X, dtype=torch.complex64)
    for s in range(sets):
        want = want + phase[s] * pattern * _fft2(sensitivities[s] * x[s])
    assert (A(x) - want).abs().max() / want.abs().max() < 1e-5
