"""The nonlinear derivative surface on a card.

Every test here needs a device and skips without one.  What they hold is that a
bundle, a Gauss-Newton step and the planner's rewrite answer on device memory
what they answer on the host, and that the encoding the fused plan claims is the
one the library's counters say ran -- a plan is not a timing, so it is read back
rather than inferred.

The point of running these at all is that ``md_`` operations take the host path
unless *every* argument is on a device, and take it silently.  A bundle member
is an assembly of a dozen small operators, each with its own scratch, so whether
one of them lands on the host is not something to infer from the source.
"""

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import linop, nlop
from bartorch.nlop.base import chain

requires_cuda = pytest.mark.skipif(
    not bartorch._cuda.available(), reason="no CUDA device, or the library was built without CUDA"
)

SHAPE = (6,)
COILS, N = 4, 16


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _elementwise():
    return {
        "exp": nlop.Exp(SHAPE),
        "log": nlop.Log(SHAPE),
        "inverse": nlop.Inverse(SHAPE),
        "abs": nlop.Abs(SHAPE),
        "multiply": nlop.Multiply((1, 6), (3, 6)),
    }


# --- a bundle on device memory -------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("which", sorted(_elementwise()))
def test_a_bundle_answers_on_the_card_what_it_answers_on_the_host(which):
    torch.manual_seed(0)
    op = _elementwise()[which]
    xs = [_rand(*shape) + 3.0 for shape in op.ishapes]
    dxs = [_rand(*shape) for shape in op.ishapes]
    dzs = [_rand(*shape) for shape in op.oshapes]

    host = op.bundle.derivative(*dxs, *xs)
    device = op.bundle.derivative(*(t.cuda() for t in dxs), *(t.cuda() for t in xs))
    assert device.device.type == "cuda"
    torch.testing.assert_close(device.cpu(), host, rtol=1e-4, atol=1e-5)

    def back(tensors):
        made = op.bundle.adjoint(*tensors)
        return (made,) if isinstance(made, torch.Tensor) else made

    on_host = back([*dzs, *xs])
    on_card = back([t.cuda() for t in (*dzs, *xs)])
    for one, other in zip(on_card, on_host):
        assert one.device.type == "cuda"
        torch.testing.assert_close(one.cpu(), other, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_a_composed_bundle_answers_on_the_card():
    """The chain rule recomputes the intermediate point, so the recomputation runs there too."""
    torch.manual_seed(0)
    op = chain(nlop.Exp(SHAPE), nlop.Multiply(SHAPE, SHAPE), output=0, input=1)
    xs = [_rand(*shape) + 3.0 for shape in op.ishapes]
    dxs = [_rand(*shape) for shape in op.ishapes]

    host = op.bundle.derivative(*dxs, *xs)
    device = op.bundle.derivative(*(t.cuda() for t in dxs), *(t.cuda() for t in xs))
    torch.testing.assert_close(device.cpu(), host, rtol=1e-4, atol=1e-5)


# --- a step on device memory ---------------------------------------------------


def _bilinear_step(iterations=2):
    schedule = nlop.IRGNM(iterations=iterations, alpha=1.0, redu=2.0, cg_maxiter=30, cg_tol=0.0)
    return schedule.operator(nlop.Multiply((1, 4), (3, 4)))


@requires_cuda
def test_a_gauss_newton_step_answers_on_the_card():
    torch.manual_seed(0)
    step = _bilinear_step()
    y = _rand(*step.data_shape)
    xn = _rand(*step.state_shape) * 0.3 + 1.0
    x0 = _rand(*step.state_shape) * 0.3

    host = step(y, xn, x0, 1.0)
    device = step(y.cuda(), xn.cuda(), x0.cuda(), step.weight(1.0, device="cuda"))
    assert device.device.type == "cuda"
    torch.testing.assert_close(device.cpu(), host, rtol=1e-3, atol=1e-4)


@requires_cuda
def test_a_step_on_the_card_carries_a_gradient_there():
    torch.manual_seed(0)
    step = _bilinear_step(iterations=1)
    y = _rand(*step.data_shape).cuda()
    x0 = (_rand(*step.state_shape) * 0.3).cuda()
    iterate = (_rand(*step.state_shape) * 0.3 + 1.0).cuda().requires_grad_(True)

    step(y, iterate, x0, step.weight(1.0, device="cuda")).abs().square().sum().backward()
    assert iterate.grad.device.type == "cuda"
    assert torch.isfinite(iterate.grad).all()
    assert iterate.grad.abs().max() > 0


# --- the plan, on the card -----------------------------------------------------


def _coil_model(off_grid, device=None):
    """The coil composition, with whatever the encoding holds put where ``device`` says."""
    maps_shape = (COILS, 1, N, N)
    if not off_grid:
        return nlop.CoilSense(linop.FFT(maps_shape, axes=(-1, -2)))
    traj = bt.traj(x=N, y=21)
    return nlop.CoilSense(linop.NUFFT(traj if device is None else traj.to(device), maps_shape))


@requires_cuda
@pytest.mark.parametrize("off_grid", [False, True])
def test_the_fused_plan_is_taken_on_the_card_too(off_grid):
    """Built from what the encoding holds on the card, so it is that operator's plan."""
    model = _coil_model(off_grid, device="cuda")
    step = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0).operator(model)
    assert step.plan.fused
    assert "normal" == step.plan.domain


@requires_cuda
def test_a_fused_coil_step_answers_on_the_card_what_it_answers_on_the_host():
    torch.manual_seed(0)
    model = _coil_model(off_grid=False)
    step = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=20, cg_tol=0.0).operator(model)

    kspace = _rand(*model.oshapes[0])
    start = _rand(*step.state_shape) * 0.2 + 1.0

    host = step(step.prepare(kspace), start, start, 1.0)
    device = step(
        step.prepare(kspace.cuda()),
        start.cuda(),
        start.cuda(),
        step.weight(1.0, device="cuda"),
    )
    assert device.device.type == "cuda"
    torch.testing.assert_close(device.cpu(), host, rtol=1e-3, atol=1e-4)


@requires_cuda
def test_the_normal_a_fused_step_applies_off_the_grid_is_the_point_spread_functions():
    """``bartorch._finufft``'s counters say which normal was built, rather than a timing."""
    from bartorch import _finufft

    torch.manual_seed(0)
    _finufft.reset_counters()
    model = _coil_model(off_grid=True, device="cuda")
    step = nlop.IRGNM(iterations=1, cg_maxiter=10, cg_tol=0.0).operator(model)

    kspace = _rand(*model.oshapes[0]).cuda()
    start = (_rand(*step.state_shape) * 0.2 + 1.0).cuda()
    step(step.prepare(kspace), start, start, step.weight(1.0, device="cuda"))

    by_psf, by_pair = _finufft.normals_built()
    assert by_psf > 0, f"the step applied the pair {by_pair} times and the function {by_psf}"
    assert 0 == _finufft.operators_built()[1], "BART's own gridder built an operator"


@requires_cuda
def test_the_two_domains_agree_on_the_card_as_they_do_on_the_host():
    torch.manual_seed(0)
    model = _coil_model(off_grid=False)
    schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=20, cg_tol=0.0)
    fused, paired = schedule.operator(model), schedule.operator(model, fuse=False)

    kspace = _rand(*model.oshapes[0]).cuda()
    start = (_rand(*fused.state_shape) * 0.2 + 1.0).cuda()
    weight = fused.weight(1.0, device="cuda")

    one = fused(fused.prepare(kspace), start, start, weight)
    other = paired(paired.prepare(kspace), start, start, weight)
    assert (one - other).abs().max() < 1e-3 * other.abs().max()


# --- the surface the retirement of _Cell rests on ------------------------------


@requires_cuda
def test_a_diagonal_set_on_the_card_is_what_the_operator_applies():
    """``multiplace`` moves the values to wherever the operator lives."""
    from bartorch.linop.basic import Sampling

    torch.manual_seed(0)
    shape = (2, 8, 8)
    first = (torch.rand(1, 8, 8) > 0.4).to(torch.complex64).cuda()
    second = (torch.rand(1, 8, 8) > 0.4).to(torch.complex64).cuda()
    x = _rand(*shape).cuda()

    A = Sampling(first, shape)
    gram = A.gram()
    torch.testing.assert_close(A(x), x * first)

    A.set(second)
    assert A(x).device.type == "cuda"
    torch.testing.assert_close(A(x), x * second)
    torch.testing.assert_close(gram(x), x * second.abs() ** 2, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_a_step_on_the_card_answers_for_a_pattern_set_after_assembly():
    from bartorch.linop.basic import Sampling

    torch.manual_seed(0)
    coils, n = 4, 16
    shape = (coils, n, n)
    schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=15, cg_tol=0.0)
    kspace = _rand(*shape).cuda()

    def assembled(pattern):
        sampling = Sampling(pattern, shape)
        model = nlop.CoilSense(sampling @ linop.FFT(shape, axes=(-1, -2)))
        return sampling, schedule.operator(model)

    first = (torch.rand(1, n, n) > 0.3).to(torch.complex64).cuda()
    second = (torch.rand(1, n, n) > 0.3).to(torch.complex64).cuda()

    def solve(step):
        state = step.start(device="cuda")
        return step(step.prepare(kspace), state, state, step.weight(1.0, device="cuda"))

    sampling, step = assembled(first)
    solve(step)
    sampling.set(second)
    reused = solve(step)

    _, rebuilt = assembled(second)
    assert reused.device.type == "cuda"
    assert torch.equal(reused, solve(rebuilt))


@requires_cuda
def test_a_batched_step_on_the_card_answers_what_each_item_answers_alone():
    """``nlop_stack_multiple`` is given ``multigpu = 0``, so the stack stays on one card."""
    torch.manual_seed(0)
    batch = 3
    schedule = nlop.IRGNM(iterations=2, alpha=1.0, redu=2.0, cg_maxiter=15, cg_tol=0.0)
    model = nlop.Multiply((1, 4), (3, 4))
    one = schedule.operator(model)
    many = schedule.operator(model, batch=batch)

    data = _rand(batch, *one.data_shape).cuda()
    state = (_rand(batch, *one.state_shape) * 0.3 + 1.0).cuda()

    weight = one.weight(1.0, device="cuda")
    alone = torch.stack([one(data[i], state[i], state[i], weight) for i in range(batch)])
    together = many(data, state, state, many.weight(1.0, device="cuda"))
    assert together.device.type == "cuda"
    assert torch.equal(alone, together)
