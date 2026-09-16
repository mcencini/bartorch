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


STATE = 16


def _bilinear():
    return nlop.Multiply((1, 4), (3, 4))


def _stepped(block, F, y, xn, x0, iterations):
    state = block.start(y, F, x0=xn, xref=x0)
    for _ in range(iterations):
        state = block(state, F)
    return state.x


@requires_cuda
def test_a_gauss_newton_step_answers_on_the_card():
    torch.manual_seed(0)
    F, block = _bilinear(), nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=30)
    y = _rand(3, 4)
    xn = _rand(STATE) * 0.3 + 1.0
    x0 = _rand(STATE) * 0.3

    host = _stepped(block, F, y, xn, x0, 2)
    device = _stepped(block, F, y.cuda(), xn.cuda(), x0.cuda(), 2)
    assert device.device.type == "cuda"
    torch.testing.assert_close(device.cpu(), host, rtol=1e-3, atol=1e-4)


@requires_cuda
def test_a_step_on_the_card_carries_a_gradient_there():
    torch.manual_seed(0)
    F, block = _bilinear(), nlop.IRGNMBlock(cg_maxiter=30)
    y = _rand(3, 4).cuda()
    x0 = (_rand(STATE) * 0.3).cuda()
    iterate = (_rand(STATE) * 0.3 + 1.0).cuda().requires_grad_(True)

    _stepped(block, F, y, iterate, x0, 1).abs().square().sum().backward()
    assert iterate.grad.device.type == "cuda"
    assert torch.isfinite(iterate.grad).all()
    assert iterate.grad.abs().max() > 0


@requires_cuda
def test_an_inner_solver_on_the_card_carries_a_gradient_there():
    """The second form: the linearization and its conjugate gradients on device memory."""
    from bartorch import optim

    torch.manual_seed(0)
    F = _bilinear()
    block = nlop.IRGNMBlock(inner=optim.CG(maxiter=30))
    y = _rand(3, 4)
    x0 = _rand(STATE) * 0.3
    host = _stepped(block, F, y, x0 + 1.0, x0, 2)

    iterate = (x0 + 1.0).cuda().requires_grad_(True)
    device = _stepped(block, F, y.cuda(), iterate, x0.cuda(), 2)
    torch.testing.assert_close(device.detach().cpu(), host, rtol=1e-3, atol=1e-4)
    device.abs().square().sum().backward()
    assert iterate.grad.device.type == "cuda"
    assert torch.isfinite(iterate.grad).all()


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
    plan = nlop.IRGNMBlock().plan(_coil_model(off_grid, device="cuda"))
    assert plan.fused
    assert "normal" == plan.domain


def _start(model):
    from bartorch.nlop.step import Linearized

    return _rand(*Linearized(model).state_shape) * 0.2 + 1.0


@requires_cuda
def test_a_fused_coil_step_answers_on_the_card_what_it_answers_on_the_host():
    torch.manual_seed(0)
    model = _coil_model(off_grid=False)
    block = nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=20)
    kspace = _rand(*model.oshapes[0])
    start = _start(model)

    host = _stepped(block, model, kspace, start, start, 2)
    device = _stepped(block, model, kspace.cuda(), start.cuda(), start.cuda(), 2)
    assert device.device.type == "cuda"
    torch.testing.assert_close(device.cpu(), host, rtol=1e-3, atol=1e-4)


@requires_cuda
def test_the_normal_a_fused_step_applies_off_the_grid_is_the_point_spread_functions():
    """``bartorch._finufft``'s counters say which normal was built, rather than a timing."""
    from bartorch import _finufft

    torch.manual_seed(0)
    _finufft.reset_counters()
    model = _coil_model(off_grid=True, device="cuda")
    block = nlop.IRGNMBlock(cg_maxiter=10)
    kspace = _rand(*model.oshapes[0]).cuda()
    start = _start(model).cuda()
    _stepped(block, model, kspace, start, start, 1)

    by_psf, by_pair = _finufft.normals_built()
    assert by_psf > 0, f"the step applied the pair {by_pair} times and the function {by_psf}"
    assert 0 == _finufft.operators_built()[1], "BART's own gridder built an operator"


@requires_cuda
def test_the_two_domains_agree_on_the_card_as_they_do_on_the_host():
    torch.manual_seed(0)
    model = _coil_model(off_grid=False)
    kspace = _rand(*model.oshapes[0]).cuda()
    start = _start(model).cuda()
    one, other = (
        _stepped(nlop.IRGNMBlock(cg_maxiter=20, fuse=fuse), model, kspace, start, start, 2)
        for fuse in (True, False)
    )
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
def test_a_step_on_the_card_answers_for_a_pattern_set_after_the_model_was_prepared():
    from bartorch.linop.basic import Sampling

    torch.manual_seed(0)
    coils, n = 4, 16
    shape = (coils, n, n)
    kspace = _rand(*shape).cuda()

    def sampled(pattern):
        sampling = Sampling(pattern, shape)
        return sampling, nlop.CoilSense(sampling @ linop.FFT(shape, axes=(-1, -2)))

    first = (torch.rand(1, n, n) > 0.3).to(torch.complex64).cuda()
    second = (torch.rand(1, n, n) > 0.3).to(torch.complex64).cuda()

    def solve(block, model):
        state = block.start(kspace, model)
        for _ in range(2):
            state = block(state, model)
        return state.x

    block = nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=15)
    sampling, model = sampled(first)
    solve(block, model)
    sampling.set(second)
    reused = solve(block, model)

    _, rebuilt = sampled(second)
    assert reused.device.type == "cuda"
    assert torch.equal(reused, solve(nlop.IRGNMBlock(cg_maxiter=15), rebuilt))


@requires_cuda
def test_a_batch_on_the_card_answers_what_each_item_answers_alone():
    torch.manual_seed(0)
    batch = 3
    F, block = _bilinear(), nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=15)
    data = _rand(batch, 3, 4).cuda()
    state = (_rand(batch, STATE) * 0.3 + 1.0).cuda()

    alone = torch.stack([_stepped(block, F, data[i], state[i], state[i], 2) for i in range(batch)])
    together = _stepped(block, F, data, state, state, 2)
    assert together.device.type == "cuda"
    assert torch.equal(alone, together)
