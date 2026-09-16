"""BART's Gauss-Newton step, taken by IRGNMBlock over any model's bundle.

The block is held against a Gauss-Newton loop written out in torch, whose inner
problem is solved exactly rather than by conjugate gradients -- so what is
compared is the method and not two paths through the same iteration.  What
pins the noir composition is the reconstruction in
``tests/test_nlop_newton.py``, measured against a phantom rather than BART.
"""

import pytest
import torch

from bartorch import linop, nlop
from bartorch.nlop.basic import Multiply
from bartorch.nlop.step import Linearized

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


def _model():
    return Multiply(IMAGE, COILS)


def stepped(block, F, y, xn, x0, iterations):
    """``iterations`` steps from ``xn`` towards the centre ``x0``; the state it reaches."""
    state = block.start(y, F, x0=xn, xref=x0)
    for _ in range(iterations):
        state = block(state, F)
    return state.x


def _arguments(generator):
    return (
        rand((3, 4), generator),
        rand((STATE,), generator) * 0.3 + 1.0,
        rand((STATE,), generator) * 0.3,
    )


# --- what the block takes and returns ----------------------------------------


def test_the_state_is_the_unknowns_laid_end_to_end(generator):
    space = Linearized(_model())
    image, coils = rand(IMAGE, generator), rand(COILS, generator)
    state = space.join(image, coils)
    assert (STATE,) == tuple(state.shape)
    back = space.split(state)
    assert torch.equal(back[0], image)
    assert torch.equal(back[1], coils)


def test_a_start_is_a_state_or_the_unknowns(generator):
    F, block = _model(), nlop.IRGNMBlock()
    y, xn, _ = _arguments(generator)
    laid = block.start(y, F, x0=xn)
    apart = block.start(y, F, x0=_split(xn))
    assert torch.equal(laid.x, apart.x)
    assert torch.equal(laid.xref, torch.zeros(STATE, dtype=torch.complex64))


def test_the_output_is_one_tensor_per_unknown(generator):
    F, block = _model(), nlop.IRGNMBlock(cg_maxiter=10)
    y, xn, x0 = _arguments(generator)
    state = block(block.start(y, F, x0=xn, xref=x0), F)
    image, coils = block.output(state, F)
    assert (IMAGE, COILS) == (tuple(image.shape), tuple(coils.shape))


def test_alpha_min0_is_refused_without_an_inner_solver():
    with pytest.raises(ValueError, match="second form"):
        nlop.IRGNMBlock(alpha_min0=0.1)


# --- against a loop written out outside BART ---------------------------------


@pytest.mark.parametrize(
    ("iterations", "alpha", "redu"), [(1, 1.0, 2.0), (3, 1.0, 2.0), (4, 0.5, 3.0)]
)
def test_the_block_is_the_method_written_out(iterations, alpha, redu, generator):
    block = nlop.IRGNMBlock(alpha=alpha, redu=redu, cg_maxiter=200)
    y, xn, x0 = _arguments(generator)
    got = stepped(block, _model(), y, xn, x0, iterations)
    weight = torch.full((STATE,), alpha, dtype=torch.complex64)
    want = written_out(y, xn, x0, weight, iterations=iterations, redu=redu)
    assert (got - want).abs().max() < 1e-4 * want.abs().max()


def test_the_weight_decays_towards_its_floor(generator):
    block = nlop.IRGNMBlock(redu=2.0, alpha_min=0.25, cg_maxiter=200)
    y, xn, x0 = _arguments(generator)
    got = stepped(block, _model(), y, xn, x0, 3)
    weight = torch.ones(STATE, dtype=torch.complex64)
    want = written_out(y, xn, x0, weight, iterations=3, redu=2.0, alpha_min=0.25)
    assert (got - want).abs().max() < 1e-4 * want.abs().max()


@pytest.mark.parametrize("at", ["y", "xn", "x0", "alpha"])
def test_every_argument_carries_the_gradient_the_written_out_loop_does(at, generator):
    """Including the second-order terms, which the implicit solve is what supplies."""
    names = ("y", "xn", "x0")
    base = _arguments(generator)
    seed = rand((STATE,), generator)

    block = nlop.IRGNMBlock(cg_maxiter=200)
    mine = [one.clone() for one in base]
    if "alpha" == at:
        block.alpha.requires_grad_(True)
    else:
        mine[names.index(at)].requires_grad_(True)
    (stepped(block, _model(), *mine, 2).conj() * seed).sum().real.backward()

    theirs = [one.clone() for one in base]
    alpha = torch.tensor(1.0, dtype=torch.float64, requires_grad="alpha" == at)
    if "alpha" != at:
        theirs[names.index(at)].requires_grad_(True)
    weight = alpha.float().to(torch.complex64) * torch.ones(STATE, dtype=torch.complex64)
    (written_out(*theirs, weight, iterations=2).conj() * seed).sum().real.backward()

    if "alpha" == at:
        got, want = block.alpha.grad, alpha.grad
    else:
        got, want = mine[names.index(at)].grad, theirs[names.index(at)].grad
    assert (got - want).abs().max() < 1e-4 * want.abs().max()


def _stack(blocks, F, y, xn, x0):
    state = blocks[0].start(y, F, x0=xn, xref=x0)
    for block in blocks:
        state = block(state, F)
    return state.x


def test_a_stack_of_blocks_with_one_alpha_is_the_schedule_to_the_bit(generator):
    y, xn, x0 = _arguments(generator)
    F = _model()
    looped = stepped(nlop.IRGNMBlock(cg_maxiter=30), F, y, xn, x0, 3)
    stacked = _stack([nlop.IRGNMBlock(cg_maxiter=30) for _ in range(3)], F, y, xn, x0)
    schedule = nlop.IRGNM(iterations=3, cg_maxiter=30)(y, F, x0=xn, xref=x0)
    assert torch.equal(looped, stacked)
    image, coils = schedule
    assert torch.equal(looped, torch.cat([image.reshape(-1), coils.reshape(-1)]))


def test_a_block_at_step_k_applies_its_own_alpha_decayed_k_times(generator):
    y, xn, x0 = _arguments(generator)
    F = _model()
    blocks = [
        nlop.IRGNMBlock(alpha=0.8, cg_maxiter=200),
        nlop.IRGNMBlock(alpha=3.0, cg_maxiter=200),
    ]
    got = _stack(blocks, F, y, xn, x0)
    first = written_out(y, xn, x0, torch.full((STATE,), 0.8, dtype=torch.complex64), iterations=1)
    want = written_out(y, first, x0, torch.full((STATE,), 1.5, dtype=torch.complex64), iterations=1)
    assert (got - want).abs().max() < 1e-4 * want.abs().max()


def test_learned_weights_are_one_per_block(generator):
    y, xn, x0 = _arguments(generator)
    F = _model()
    frozen = _stack([nlop.IRGNMBlock(cg_maxiter=30) for _ in range(3)], F, y, xn, x0)
    blocks = [nlop.IRGNMBlock(cg_maxiter=30) for _ in range(3)]
    for block in blocks:
        block.alpha.requires_grad_(True)
    learned = _stack(blocks, F, y, xn, x0)
    assert torch.equal(learned.detach(), frozen)

    learned.abs().square().sum().backward()
    grads = [block.alpha.grad.item() for block in blocks]
    assert all(0 != g for g in grads)
    assert len(set(grads)) == len(grads)


def test_a_model_of_one_unknown_is_written_flat_too():
    """What the step asserts is the state's rank, not how many unknowns made it."""
    space = Linearized(linop.FFT((2, 3), axes=(-1,)).to_nonlinear())
    assert space.state_shape == (6,)
    assert space.split(torch.zeros(6, dtype=torch.complex64))[0].shape == (2, 3)


def test_the_block_reaches_a_model_built_from_torch(generator):
    made = nlop.TorchOperator(lambda p: p * torch.exp(-p), (5,), (5,))
    block = nlop.IRGNMBlock(cg_maxiter=40)
    y = rand((5,), generator)
    x0 = rand((5,), generator) * 0.1 + 1.0
    state = block(block.start(y, made, x0=x0, xref=x0), made)
    assert (5,) == tuple(block.output(state, made).shape)


def test_a_model_without_a_bundle_is_refused():
    """The library's own inverse is built in C and declares no derivative of the point."""
    with pytest.raises(TypeError, match="no derivative as a function of the point"):
        Linearized(Linearized(_model()).inverse())


# --- the plan the step took --------------------------------------------------


def _coil_model(n=16, coils=4, off_grid=False):
    import bartorch.tools as bt

    shape = (coils, 1, n, n)
    if off_grid:
        return nlop.CoilSense(linop.NUFFT(bt.traj(x=n, y=21), shape))
    return nlop.CoilSense(linop.FFT(shape, axes=(-1, -2)))


@pytest.mark.parametrize("off_grid", [False, True])
def test_a_coil_composition_is_lowered_into_the_normal_equation_domain(off_grid):
    plan = nlop.IRGNMBlock().plan(_coil_model(off_grid=off_grid))
    assert plan.fused
    assert "normal" == plan.domain
    assert "chain rule" == plan.derivative
    # The data is what the encoding's adjoint returns, not what it takes.
    assert (4, 1, 16, 16) == Linearized(_coil_model(off_grid=off_grid)).data_shape


@pytest.mark.parametrize("off_grid", [False, True])
def test_the_rewrite_is_declined_when_it_is_declined(off_grid):
    plan = nlop.IRGNMBlock(fuse=False).plan(_coil_model(off_grid=off_grid))
    assert not plan.fused
    assert "paired" == plan.domain


def test_a_model_that_is_not_a_coil_composition_has_nothing_to_lower():
    plan = nlop.IRGNMBlock().plan(_model())
    assert not plan.fused
    assert plan.encoding is None
    assert "declared" == plan.derivative


def test_preparing_the_data_is_the_encodings_adjoint(generator):
    encoding = linop.FFT((4, 1, 16, 16), axes=(-1, -2))
    F = nlop.CoilSense(encoding)
    kspace = rand((4, 1, 16, 16), generator)
    assert torch.equal(nlop.IRGNMBlock().start(kspace, F).data, encoding.adjoint(kspace))


def _fused_and_paired(F, kspace, start):
    answers = []
    for fuse in (True, False):
        block = nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=60, fuse=fuse)
        answers.append(stepped(block, F, kspace, start, start, 2))
    return answers


def test_the_grid_leaves_the_fused_and_the_paired_answers_agreeing(generator):
    """On a grid the two are the same arithmetic up to single precision."""
    F = _coil_model()
    kspace = rand(tuple(F.oshape), generator)
    start = rand(Linearized(F).state_shape, generator) * 0.2 + 1.0
    one, other = _fused_and_paired(F, kspace, start)
    assert (one - other).abs().max() < 1e-4 * other.abs().max()


def test_off_the_grid_the_two_close_with_the_transforms_tolerance():
    """Which says what separates them is the NUFFT's accuracy and not the rewrite."""
    from bartorch import _finufft

    distances = []
    for tolerance in (1e-2, 1e-3, 1e-5):
        _finufft.use_in_tools(tolerance=tolerance)
        F = _coil_model(off_grid=True)
        state = torch.Generator().manual_seed(4)
        kspace = rand(tuple(F.oshape), state)
        start = rand(Linearized(F).state_shape, state) * 0.2 + 1.0
        one, other = _fused_and_paired(F, kspace, start)
        distances.append(((one - other).abs().max() / other.abs().max()).item())

    assert distances[0] > distances[1] > distances[2]
    assert distances[2] < 1e-3


@pytest.mark.parametrize("off_grid", [False, True])
def test_an_application_leaves_the_derivative_available_after_a_shared_solve(off_grid):
    """The inverse's backward pass selects only some derivatives on nodes it shares
    with the model; the next application of the model has to select them all again."""
    space = Linearized(_coil_model(off_grid=off_grid), cg_maxiter=10)
    point = rand(space.state_shape, torch.Generator().manual_seed(0)) * 0.2 + 1.0

    space.inverse().forward(point, point, torch.ones_like(point))
    space.inverse()._jacobian(0, 1).adjoint(point)

    value = space.operator.forward(point)
    assert torch.isfinite(space.operator._adjoint(value)).all()


# --- BART's own model ---------------------------------------------------------


def test_barts_noir_model_arrives_lowered_off_the_grid_and_not_on_it():
    """``noir2_join``'s own choice, read back through the composition it writes."""
    import bartorch.tools as bt
    from bartorch.nlop.plan import describe

    assert not describe(nlop.CartesianSense((4, 16, 16))).lowered
    assert describe(nlop.NoncartesianSense(bt.traj(x=16, y=21), (4, 16, 16))).lowered


def test_the_planner_lowers_the_model_bart_left_paired():
    """Which is the ground the planner adds: on a grid BART applies the pair."""
    model = nlop.CartesianSense((4, 16, 16))._composition()
    assert nlop.IRGNMBlock().plan(model).fused
    assert "normal" == nlop.IRGNMBlock().plan(model).domain
    assert not nlop.IRGNMBlock(fuse=False).plan(model).fused


# --- a pattern rewritten under a prepared model -------------------------------


def _sampled(pattern, shape):
    """A coil model whose encoding carries ``pattern``, and its sampling operator."""
    from bartorch.linop.basic import Sampling

    sampling = Sampling(pattern, shape)
    return sampling, nlop.CoilSense(sampling @ linop.FFT(shape, axes=(-1, -2)))


def test_a_block_answers_for_a_pattern_set_after_the_model_was_prepared():
    """A new mask costs no rebuild, which is the reuse BART's noir model had.

    Held against a block over a model built with the second pattern from the
    start, which is the answer the caller would have got by rebuilding.
    """
    torch.manual_seed(0)
    coils, n = 4, 16
    shape = (coils, n, n)
    first = (torch.rand(1, n, n) > 0.3).to(torch.complex64)
    second = (torch.rand(1, n, n) > 0.3).to(torch.complex64)
    kspace = torch.randn(*shape, dtype=torch.complex64)

    def solve(block, model, pattern):
        state = block.start(kspace * pattern, model)
        for _ in range(2):
            state = block(state, model)
        return state.x

    block = nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=15)
    sampling, model = _sampled(first, shape)
    on_first = solve(block, model, first)

    sampling.set(second)
    reused = solve(block, model, second)

    _, rebuilt = _sampled(second, shape)
    assert not torch.allclose(on_first, reused), "the swap changed nothing"
    assert torch.equal(reused, solve(nlop.IRGNMBlock(cg_maxiter=15), rebuilt, second))


# --- a batch of independent items ----------------------------------------------


def test_a_batch_answers_what_each_item_answers_alone():
    """The claim the batch makes: items share nothing, not even the inner solve.

    Conjugate gradients couple through global inner products, so a batch laid
    into one state would *not* answer this; the block takes each item's step on
    its own.
    """
    torch.manual_seed(0)
    batch = 4
    F, block = _model(), nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=15)
    data = torch.randn(batch, 3, 4, dtype=torch.complex64)
    start = torch.randn(batch, STATE, dtype=torch.complex64) * 0.3 + 1.0

    together = stepped(block, F, data, start, start, 2)
    assert (batch, STATE) == tuple(together.shape)
    alone = torch.stack([stepped(block, F, data[i], start[i], start[i], 2) for i in range(batch)])
    assert torch.equal(alone, together)


def test_a_batched_state_splits_and_joins_with_the_batch_in_front():
    torch.manual_seed(0)
    space = Linearized(_model())
    state = torch.randn(3, STATE, dtype=torch.complex64)
    parts = space.split(state)
    assert [(3, *shape) for shape in space.lowered.ishapes] == [tuple(p.shape) for p in parts]
    assert torch.equal(space.join(*parts), state)


def test_a_batch_carries_a_gradient():
    torch.manual_seed(0)
    F, block = _model(), nlop.IRGNMBlock(cg_maxiter=10)
    data = torch.randn(2, 3, 4, dtype=torch.complex64)
    centre = torch.randn(2, STATE, dtype=torch.complex64) * 0.3
    iterate = (torch.randn(2, STATE, dtype=torch.complex64) * 0.3 + 1.0).requires_grad_(True)
    stepped(block, F, data, iterate, centre, 1).abs().square().sum().backward()
    assert torch.isfinite(iterate.grad).all()
    assert 0 < iterate.grad.abs().max()


# --- a batch of items in one model ------------------------------------------------


def _phantoms(count, n=16, coils=4):
    """An image and smooth coils per item, each a little different."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    centres = ((-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5))[:coils]
    profiles = torch.stack([torch.exp(-((xx - s) ** 2 + (yy - t) ** 2)) for s, t in centres])
    images, maps = [], []
    for i in range(count):
        images.append(((xx**2 + yy**2) < 0.3 + 0.1 * i).to(torch.complex64).reshape(1, 1, n, n))
        maps.append((profiles * (1.0 + 0.2 * i)).to(torch.complex64).reshape(coils, 1, n, n))
    return images, maps


def _encodings(off_grid, n=16):
    import bartorch.tools as bt

    if off_grid:
        trajectory = bt.traj(x=n, y=21)
        return lambda shape: linop.NUFFT(trajectory, shape)
    return lambda shape: linop.FFT(shape, axes=(-1, -2))


def test_a_model_of_items_leads_every_shape_with_them():
    F = nlop.CoilSense(linop.FFT((3, 4, 1, 8, 8), axes=(-1, -2)), items=True)
    assert 3 == F.items
    assert ((3, 1, 1, 8, 8), (3, 4, 1, 8, 8)) == F.ishapes
    space = Linearized(F)
    assert (3, 64 + 256) == space.state_shape
    image, coils = space.split(space.start())
    assert torch.equal(image, torch.ones(3, 1, 1, 8, 8, dtype=torch.complex64))
    assert torch.equal(coils, torch.zeros(3, 4, 1, 8, 8, dtype=torch.complex64))
    assert torch.equal(space.join(image, coils), space.start())


def test_items_that_disagree_on_their_count_are_refused():
    with pytest.raises(ValueError, match="items lead every shape"):
        nlop.CoilSense(
            linop.FFT((3, 4, 1, 8, 8), axes=(-1, -2)), image_shape=(2, 1, 1, 8, 8), items=True
        )


@pytest.mark.parametrize("off_grid", [False, True])
def test_a_model_of_items_steps_each_as_it_would_step_alone(off_grid):
    """One model for the batch, and each item's inner problem solved on its own."""
    items, n, coils = 3, 16, 4
    encoding = _encodings(off_grid, n)
    fused = nlop.CoilSense(encoding((items, coils, 1, n, n)), items=True)
    single = nlop.CoilSense(encoding((coils, 1, n, n)))
    kspace = torch.stack([single.forward(*pair) for pair in zip(*_phantoms(items, n, coils))])

    def run(F, y):
        block = nlop.IRGNMBlock(cg_maxiter=30)
        state = block.start(y, F)
        for _ in range(4):
            state = block(state, F)
        return state.x

    together = run(fused, kspace)
    for i in range(items):
        alone = run(single, kspace[i])
        assert (together[i] - alone).abs().max() < 1e-5 * alone.abs().max()

    # Items share nothing: another item's data moves none of them beyond the
    # transform's own reproducibility -- exact on a grid, and off it the order
    # in which FINUFFT's threads spread, which varies between runs on some hosts.
    moved = kspace.clone()
    moved[0] *= 3.0
    others = run(fused, moved)[1:]
    floor = 1e-5 if off_grid else 1e-10
    assert (others - together[1:]).abs().max() < floor * together[1:].abs().max()


def test_a_model_of_items_carries_no_gradient_between_them():
    items, n, coils = 2, 8, 2
    encoding = _encodings(False, n)
    fused = nlop.CoilSense(encoding((items, coils, 1, n, n)), items=True)
    images, maps = _phantoms(items, n, coils)
    single = nlop.CoilSense(encoding((coils, 1, n, n)))
    kspace = torch.stack([single.forward(a, b) for a, b in zip(images, maps)])
    kspace.requires_grad_(True)

    block = nlop.IRGNMBlock(cg_maxiter=30)
    state = block.start(kspace, fused)
    for _ in range(2):
        state = block(state, fused)
    state.x[1].abs().square().sum().backward()
    assert (kspace.grad[0].abs().max() < 1e-6 * kspace.grad[1].abs().max()).item()
    assert (kspace.grad[1] != 0).any()


def test_an_inner_solver_over_a_model_of_items_is_refused():
    from bartorch import optim

    F = nlop.CoilSense(linop.FFT((2, 2, 1, 8, 8), axes=(-1, -2)), items=True)
    block = nlop.IRGNMBlock(inner=optim.CG())
    with pytest.raises(ValueError, match="several items"):
        block.start(torch.zeros(F.oshapes[0], dtype=torch.complex64), F)
