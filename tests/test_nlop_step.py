"""BART's Gauss-Newton step assembled over any model's bundle.

The step is held against a Gauss-Newton loop written out in torch, whose inner
problem is solved exactly rather than by conjugate gradients -- so what is
compared is the method and not two paths through the same iteration.  The
comparison with ``_Cell`` is BART against BART and says only that the assembly
here and BART's own are the same expression.
"""

import pytest
import torch

from bartorch import linop, nlop
from bartorch.linop.basic import Identity
from bartorch.nlop.base import chain
from bartorch.nlop.basic import Multiply
from bartorch.nlop.bundle import Asymmetric
from bartorch.nlop.step import Step

#: A bilinear model small enough to write its Jacobian out as a matrix.
IMAGE, COILS = (1, 4), (3, 4)
SIZES = (4, 12)
STATE = sum(SIZES)


def rand(shape, generator):
    real = torch.randn(shape, generator=generator)
    imag = torch.randn(shape, generator=generator)
    return (real + 1j * imag).to(torch.complex64)


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(20260915)


def _split(x):
    return x[: SIZES[0]].reshape(IMAGE), x[SIZES[0] :].reshape(COILS)


def _jacobian(x):
    """``DF(x)`` as a dense matrix, one column per entry of the state."""
    a, b = _split(x)
    columns = []
    for at in range(STATE):
        unit = torch.zeros(STATE, dtype=torch.complex64)
        unit[at] = 1.0
        da, db = _split(unit)
        columns.append((da * b + a * db).reshape(-1))
    return torch.stack(columns, dim=1)


def written_out(y, xn, x0, alpha, *, iterations, redu=2.0, alpha_min=0.0):
    """The same schedule in torch, with each inner problem solved exactly."""
    x, weight = xn, alpha
    eye = torch.eye(STATE, dtype=torch.complex64)
    for _ in range(iterations):
        a, b = _split(x)
        jacobian = _jacobian(x)
        residual = y.reshape(-1) - (a * b).reshape(-1)
        rhs = jacobian.conj().T @ residual - weight * (x - x0)
        normal = jacobian.conj().T @ jacobian + torch.diag(weight) @ eye
        x = x + torch.linalg.solve(normal, rhs)
        weight = (weight - alpha_min) / redu + alpha_min
    return x


def _step(iterations, redu=2.0, alpha_min=0.0):
    schedule = nlop.IRGNM(
        iterations=iterations, redu=redu, alpha_min=alpha_min, cg_maxiter=200, cg_tol=0.0
    )
    return Step(nlop.Multiply(IMAGE, COILS), schedule)


def _arguments(step, generator, alpha=1.0):
    return (
        rand(step.data_shape, generator),
        rand((STATE,), generator) * 0.3 + 1.0,
        rand((STATE,), generator) * 0.3,
        step.weight(alpha),
    )


# --- what the step is --------------------------------------------------------


def test_the_step_takes_the_data_the_iterate_the_centre_and_the_weight():
    step = _step(1)
    assert step.ishapes == ((3, 4), (STATE,), (STATE,), (STATE,))
    assert step.oshapes == ((STATE,),)


def test_the_state_is_the_unknowns_laid_end_to_end(generator):
    step = _step(1)
    image, coils = rand(IMAGE, generator), rand(COILS, generator)
    state = step.join(image, coils)
    assert (STATE,) == tuple(state.shape)
    back = step.split(state)
    assert torch.equal(back[0], image)
    assert torch.equal(back[1], coils)


def test_a_weight_given_as_a_number_becomes_a_vector():
    step = _step(1)
    assert (STATE,) == tuple(step.weight(0.5).shape)
    assert torch.equal(step.weight(0.5), torch.full((STATE,), 0.5, dtype=torch.complex64))


# --- against a loop written out outside BART ---------------------------------


@pytest.mark.parametrize(
    ("iterations", "alpha", "redu"), [(1, 1.0, 2.0), (3, 1.0, 2.0), (4, 0.5, 3.0)]
)
def test_the_step_is_the_method_written_out(iterations, alpha, redu, generator):
    step = _step(iterations, redu=redu)
    y, xn, x0, weight = _arguments(step, generator, alpha)
    got = step(y, xn, x0, weight)
    want = written_out(y, xn, x0, weight, iterations=iterations, redu=redu)
    assert (got - want).abs().max() < 1e-4 * want.abs().max()


def test_the_weight_decays_towards_its_floor(generator):
    step = _step(3, redu=2.0, alpha_min=0.25)
    y, xn, x0, weight = _arguments(step, generator)
    got = step(y, xn, x0, weight)
    want = written_out(y, xn, x0, weight, iterations=3, redu=2.0, alpha_min=0.25)
    assert (got - want).abs().max() < 1e-4 * want.abs().max()


@pytest.mark.parametrize("at", [0, 1, 2, 3])
def test_every_argument_carries_the_gradient_the_written_out_loop_does(at, generator):
    """Including the second-order terms, which the implicit solve is what supplies."""
    step = _step(2)
    base = _arguments(step, generator)
    seed = rand((STATE,), generator)

    mine = [one.clone() for one in base]
    mine[at].requires_grad_(True)
    (step(*mine).conj() * seed).sum().real.backward()

    theirs = [one.clone() for one in base]
    theirs[at].requires_grad_(True)
    (written_out(*theirs, iterations=2).conj() * seed).sum().real.backward()

    assert (mine[at].grad - theirs[at].grad).abs().max() < 1e-4 * theirs[at].grad.abs().max()


# --- BART against BART -------------------------------------------------------


def _normal_domain(F):
    """``noir_get_forward``: the two unknowns multiplied, then the transform's normal."""
    image, coils, transform = F.image, F.coils, F.transform
    product = Multiply(image.oshape, coils.oshape)
    made = chain(coils.to_nonlinear(), product, output=0, input=1)
    made = chain(image.to_nonlinear(), made, output=0, input=0).permute_inputs([1, 0])
    stage = Asymmetric(transform.gram(), Identity(product.oshape))
    return chain(made, stage, output=0, input=0)


@pytest.mark.parametrize("iterations", [1, 2])
def test_the_assembly_is_barts_own_over_the_noir_composition(iterations):
    """An agreement check between two routes into BART, not a numerical test."""
    F = nlop.CartesianSense((4, 8, 8), sobolev=(220.0, 8.0), oversampling_coils=1.0)
    schedule = nlop.IRGNM(
        iterations=iterations, alpha=1.0, redu=2.0, alpha_min=0.0, cg_maxiter=30, cg_tol=0.0
    )
    cell = schedule.operator(F, batch=1)
    step = Step(_normal_domain(F), schedule)

    kspace = torch.randn(F.oshapes[0], dtype=torch.complex64)
    pattern = torch.ones((1, 1, 8, 8), dtype=torch.complex64)
    data = cell.prepare()(kspace.reshape(cell.data_shape), pattern)
    start = cell.start()

    got = step(
        data.reshape(step.data_shape),
        start.reshape(step.state_shape),
        start.reshape(step.state_shape),
        1.0,
    )
    want = cell(data, start, start, 1.0)
    assert torch.equal(got.reshape(-1), want.reshape(-1))


# --- what reaches it ---------------------------------------------------------


def test_a_model_of_one_unknown_is_written_flat_too(generator):
    """What the step asserts is the state's rank, not how many unknowns made it."""
    F = linop.FFT((2, 3), axes=(-1,)).to_nonlinear()
    step = nlop.IRGNM(iterations=1, cg_maxiter=40, cg_tol=0.0).operator(F)
    assert step.state_shape == (6,)
    assert step.split(torch.zeros(6, dtype=torch.complex64))[0].shape == (2, 3)


def test_the_step_reaches_a_model_built_from_torch(generator):
    made = nlop.FromTorch(lambda p: p * torch.exp(-p), (5,), (5,))
    step = nlop.IRGNM(iterations=1, cg_maxiter=40, cg_tol=0.0).operator(made)
    y = rand((5,), generator)
    x0 = rand((5,), generator) * 0.1 + 1.0
    assert (5,) == tuple(step(y, x0, x0, 1.0).shape)


# --- the plan the step took --------------------------------------------------


def _coil_model(n=16, coils=4, off_grid=False):
    import bartorch.tools as bt

    shape = (coils, 1, n, n)
    if off_grid:
        return nlop.CoilSense(linop.NUFFT(bt.traj(x=n, y=21), shape))
    return nlop.CoilSense(linop.FFT(shape, axes=(-1, -2)))


@pytest.mark.parametrize("off_grid", [False, True])
def test_a_coil_composition_is_lowered_into_the_normal_equation_domain(off_grid):
    step = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0).operator(
        _coil_model(off_grid=off_grid)
    )
    assert step.plan.fused
    assert "normal" == step.plan.domain
    assert "chain rule" == step.plan.bundle
    # The data is what the encoding's adjoint returns, not what it takes.
    assert (4, 1, 16, 16) == step.data_shape


@pytest.mark.parametrize("off_grid", [False, True])
def test_the_rewrite_is_declined_when_it_is_declined(off_grid):
    step = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0).operator(
        _coil_model(off_grid=off_grid), fuse=False
    )
    assert not step.plan.fused
    assert "paired" == step.plan.domain


def test_a_model_that_is_not_a_coil_composition_has_nothing_to_lower():
    step = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0).operator(nlop.Multiply(IMAGE, COILS))
    assert not step.plan.fused
    assert step.plan.encoding is None
    assert "declared" == step.plan.bundle


def test_preparing_the_data_is_the_encodings_adjoint(generator):
    encoding = linop.FFT((4, 1, 16, 16), axes=(-1, -2))
    step = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0).operator(nlop.CoilSense(encoding))
    kspace = rand((4, 1, 16, 16), generator)
    assert torch.equal(step.prepare(kspace), encoding.adjoint(kspace))


def test_the_grid_leaves_the_fused_and_the_paired_answers_agreeing(generator):
    """On a grid the two are the same arithmetic up to single precision."""
    F = _coil_model()
    schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=60, cg_tol=0.0)
    fused, paired = schedule.operator(F), schedule.operator(F, fuse=False)

    kspace = rand(paired.data_shape, generator)
    start = rand(paired.state_shape, generator) * 0.2 + 1.0
    one = fused(fused.prepare(kspace), start, start, 1.0)
    other = paired(paired.prepare(kspace), start, start, 1.0)
    assert (one - other).abs().max() < 1e-4 * other.abs().max()


def test_off_the_grid_the_two_close_with_the_transforms_tolerance(generator):
    """Which says what separates them is the NUFFT's accuracy and not the rewrite."""
    from bartorch import _finufft

    distances = []
    for tolerance in (1e-2, 1e-3, 1e-5):
        _finufft.use_in_tools(tolerance=tolerance)
        F = _coil_model(off_grid=True)
        schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=60, cg_tol=0.0)
        fused, paired = schedule.operator(F), schedule.operator(F, fuse=False)

        state = torch.Generator().manual_seed(4)
        kspace = rand(paired.data_shape, state)
        start = rand(paired.state_shape, state) * 0.2 + 1.0
        one = fused(fused.prepare(kspace), start, start, 1.0)
        other = paired(paired.prepare(kspace), start, start, 1.0)
        distances.append(((one - other).abs().max() / other.abs().max()).item())

    assert distances[0] > distances[1] > distances[2]
    assert distances[2] < 1e-3


# --- BART's own model ---------------------------------------------------------


def test_barts_noir_model_arrives_lowered_off_the_grid_and_not_on_it():
    """``noir2_join``'s own choice, read back through the composition it writes."""
    import bartorch.tools as bt
    from bartorch.nlop.plan import describe

    assert not describe(nlop.CartesianSense((4, 16, 16))).lowered
    assert describe(nlop.NoncartesianSense(bt.traj(x=16, y=21), (4, 16, 16))).lowered


def test_the_planner_lowers_the_model_bart_left_paired():
    """Which is the ground the planner adds: on a grid BART applies the pair."""
    schedule = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0)
    step = Step(nlop.CartesianSense((4, 16, 16))._composition(), schedule)
    assert step.plan.fused
    assert "normal" == step.plan.domain
    assert not Step(
        nlop.CartesianSense((4, 16, 16))._composition(), schedule, fuse=False
    ).plan.fused


# --- a pattern rewritten under a built step ------------------------------------


def _assembled(pattern, shape, schedule):
    """A step over a coil model whose encoding carries ``pattern``, and that pattern."""
    from bartorch.linop.basic import Sampling

    sampling = Sampling(pattern, shape)
    model = nlop.CoilSense(sampling @ linop.FFT(shape, axes=(-1, -2)))
    return sampling, Step(model, schedule)


def test_a_step_answers_for_a_pattern_set_after_it_was_assembled():
    """The reuse ``_Cell`` had, without its model: a new mask costs no reassembly.

    Held against a step assembled over the second pattern from the start, which
    is the answer the caller would have got by rebuilding.
    """
    torch.manual_seed(0)
    coils, n = 4, 16
    shape = (coils, n, n)
    schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=15, cg_tol=0.0)

    first = (torch.rand(1, n, n) > 0.3).to(torch.complex64)
    second = (torch.rand(1, n, n) > 0.3).to(torch.complex64)
    kspace = torch.randn(*shape, dtype=torch.complex64)

    def solve(step, pattern):
        state = torch.zeros(step.state_shape, dtype=torch.complex64)
        state[: n * n] = 1.0
        return step(step.prepare(kspace * pattern), state, state, 1.0)

    sampling, step = _assembled(first, shape, schedule)
    on_first = solve(step, first)

    sampling.set(second)
    reused = solve(step, second)

    _, rebuilt = _assembled(second, shape, schedule)
    assert not torch.allclose(on_first, reused), "the swap changed nothing"
    assert torch.equal(reused, solve(rebuilt, second))


# --- a batch of independent items ----------------------------------------------


def test_a_batched_step_answers_what_each_item_answers_alone():
    """The claim the batch makes: items share nothing, not even the inner solve.

    Conjugate gradients couple through global inner products, so a batch laid
    into one state would *not* answer this; ``nlop_stack_multiple`` builds the
    whole expression per item, which does.
    """
    torch.manual_seed(0)
    batch = 4
    schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=15, cg_tol=0.0)
    model = Multiply((1, 4), (3, 4))
    one, many = Step(model, schedule), Step(model, schedule, batch=batch)

    assert (batch, *one.state_shape) == tuple(many.state_shape)
    assert (batch, *one.data_shape) == tuple(many.data_shape)

    data = torch.randn(batch, *one.data_shape, dtype=torch.complex64)
    state = torch.randn(batch, *one.state_shape, dtype=torch.complex64) * 0.3 + 1.0

    alone = torch.stack([one(data[i], state[i], state[i], 1.0) for i in range(batch)])
    together = many(data, state, state, many.weight(1.0))
    assert torch.equal(alone, together)


def test_a_batched_state_splits_and_joins_with_the_batch_in_front():
    torch.manual_seed(0)
    schedule = nlop.IRGNM(iterations=1, cg_maxiter=5, cg_tol=0.0)
    step = Step(Multiply((1, 4), (3, 4)), schedule, batch=3)
    state = torch.randn(*step.state_shape, dtype=torch.complex64)

    parts = step.split(state)
    assert [(3, *shape) for shape in step.lowered.ishapes] == [tuple(p.shape) for p in parts]
    assert torch.equal(step.join(*parts), state)


def test_a_batched_step_carries_a_gradient():
    torch.manual_seed(0)
    schedule = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0)
    step = Step(Multiply((1, 4), (3, 4)), schedule, batch=2)
    data = torch.randn(*step.data_shape, dtype=torch.complex64)
    start = torch.randn(*step.state_shape, dtype=torch.complex64) * 0.3
    iterate = (torch.randn(*step.state_shape, dtype=torch.complex64) * 0.3 + 1.0).requires_grad_(
        True
    )

    step(data, iterate, start, step.weight(1.0)).abs().square().sum().backward()
    assert torch.isfinite(iterate.grad).all()
    assert 0 < iterate.grad.abs().max()


def test_a_batch_below_one_is_refused():
    with pytest.raises(ValueError):
        Step(Multiply((1, 4), (3, 4)), nlop.IRGNM(iterations=1), batch=0)


def test_a_batched_step_is_barts_own_batched_step_for_the_noir_model():
    """BART against BART: the assembly here and ``noir_gauss_newton_step_create``."""
    torch.manual_seed(0)
    coils, n, batch = 4, 16, 3
    model = nlop.CartesianSense((coils, n, n), sobolev=(220.0, 8.0))
    schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=20, cg_tol=0.0)

    cell = schedule.operator(model, batch=batch)
    step = Step(model, schedule, batch=batch)

    kspace = torch.randn(batch, coils, n, n, dtype=torch.complex64)
    pattern = torch.ones(batch, n, n, dtype=torch.complex64)

    start = cell.start(batch=batch)
    theirs = cell(cell.prepare()(kspace, pattern), start, start, 1.0)

    state = torch.zeros(step.state_shape, dtype=torch.complex64)
    state[:, : n * n] = 1.0
    ours = step(step.prepare(kspace * pattern.reshape(batch, 1, n, n)), state, state, 1.0)

    assert torch.equal(theirs.reshape(batch, -1), ours.reshape(batch, -1))
