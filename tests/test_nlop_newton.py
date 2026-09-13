"""BART's Gauss-Newton step, which already has a derivative.

``noir/model_net.c`` builds one iteration of ``nlinv`` as an ``nlop``, out of
``nlop``s throughout: the forward model, the derivative *as a function of the
linearisation point*, the adjoint, and ``norm_inv``'s implicitly
differentiated inverse of the normal operator.  So the step differentiates by
the data, the iterate, the regularisation centre and the weight, second-order
terms included -- which is what ``networks/nlinvnet.c`` trains through, and
what a :class:`~bartorch.nlop.GaussNewton` hands to torch.

The thing to know about the model is that it has no sampling pattern until it
is given one, and it is given one as a *side effect* of the gridding:
``noir_adjoint_fft_fun`` calls ``linop_gdiag_set_diag`` on its way past.  A
step applied before that reads a diagonal nothing has written, which is a
segmentation fault; :meth:`prepare` is what writes it, and the operator
refuses until it has been.
"""

import math

import pytest
import torch

import bartorch.tools as bt
from bartorch import linop, nlop
from bartorch._dispatch import BartError

N, COILS = 16, 2


def _phantom():
    """A disc, two smooth coils, and the k-space they make."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing="ij")
    image = ((xx**2 + yy**2) < 0.5).to(torch.complex64)
    coils = torch.stack(
        [torch.exp(-((xx - s) ** 2 + yy**2)).to(torch.complex64) for s in (-0.4, 0.4)]
    )
    F = linop.FFT((COILS, 1, N, N), axes=(-1, -2))
    return image, coils, F((coils * image).reshape(COILS, 1, N, N))


@pytest.fixture
def problem():
    torch.manual_seed(0)
    image, coils, kspace = _phantom()
    return image, coils, kspace.reshape(1, COILS, 1, N, N)


def _ones(n: int = N):
    return torch.ones(1, n, n, dtype=torch.complex64)


# --- what it takes and returns ------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
def test_the_batch_is_barts_own(batch):
    """Everywhere else here a leading axis is applied item by item from
    Python.  This is the one operator whose batch BART carries itself, because
    ``nlinvnet`` needed it."""
    newton = nlop.GaussNewton((COILS, 8, 8), batch=batch)
    assert (batch, COILS, 1, 8, 8) == newton.data_shape
    assert batch == newton.state_shape[0]
    assert 1 == len(newton.output_shapes)
    assert newton.state_shape == newton.output_shapes[0]


def test_the_short_shapes_are_the_long_ones_written_without_the_empty_axes():
    # BART's sixteen axes are what the operator records, because that is what
    # the arity check holds it to; the short form is the same memory.
    newton = nlop.GaussNewton((COILS, 8, 8), batch=2)
    assert 16 == len(newton.ishapes[0])
    assert (2, COILS, 1, 8, 8) == newton.shapes[0]
    assert math.prod(newton.ishapes[0]) == math.prod(newton.shapes[0])


def test_the_companions_say_what_they_take(problem):
    newton = nlop.GaussNewton((COILS, 8, 8))
    assert 2 == len(newton.prepare().shapes)
    assert (1, newton.state_shape[1]) == newton.decompose().shapes[0]
    assert 2 == len(newton.decompose().output_shapes)
    assert newton.join().output_shapes == (newton.state_shape,)


def test_off_the_grid_the_gridding_takes_the_trajectory_too():
    trajectory = bt.traj(x=16, y=8, r=True)
    newton = nlop.GaussNewton((COILS, 8, 8), trajectory=trajectory)
    assert 3 == len(newton.prepare().shapes)
    assert (1, *tuple(trajectory.shape)) == newton.prepare().shapes[2]


# --- the model has to be given a pattern first --------------------------------


def test_a_step_before_the_gridding_says_so_rather_than_taking_the_process(problem):
    _, _, kspace = problem
    newton = nlop.GaussNewton((COILS, N, N))
    y = torch.zeros(newton.data_shape, dtype=torch.complex64)
    x0 = newton.start()
    with pytest.raises(BartError, match="no sampling pattern yet"):
        newton(y, x0, x0, 1.0)


def test_two_operators_do_not_share_a_model(problem):
    _, _, kspace = problem
    first = nlop.GaussNewton((COILS, N, N))
    second = nlop.GaussNewton((COILS, N, N))
    y = first.prepare()(kspace, _ones())
    x0 = first.start()
    assert torch.isfinite(first(y, x0, x0, 1.0)).all()
    with pytest.raises(BartError, match="no sampling pattern yet"):
        second(y, x0, x0, 1.0)


# --- the iterate ---------------------------------------------------------------


def test_the_start_is_an_image_of_ones_and_no_coils():
    newton = nlop.GaussNewton((COILS, 8, 8))
    image, coils = newton.split()(newton.start())
    torch.testing.assert_close(image, torch.ones_like(image), rtol=0, atol=0)
    torch.testing.assert_close(coils, torch.zeros_like(coils), rtol=0, atol=0)


def test_split_and_join_are_each_other():
    newton = nlop.GaussNewton((COILS, 8, 8))
    x = torch.randn(newton.state_shape, dtype=torch.complex64)
    image, coils = newton.split()(x)
    torch.testing.assert_close(newton.join()(image, coils), x, rtol=0, atol=0)


def test_decompose_carries_the_transforms_and_split_does_not():
    # `decompose` runs the coils through the Sobolev weighting, so what comes
    # back is profiles rather than the coefficients that were fitted.
    newton = nlop.GaussNewton((COILS, 8, 8))
    x = torch.randn(newton.state_shape, dtype=torch.complex64)
    assert not torch.equal(newton.decompose()(x)[1], newton.split()(x)[1])


# --- it reconstructs ------------------------------------------------------------


def test_it_recovers_the_image_as_well_as_the_tool_does(problem):
    image, _, kspace = problem
    newton = nlop.GaussNewton((COILS, N, N), iterations=8, redu=2.0)
    y = newton.prepare()(kspace, _ones())
    x0 = newton.start()
    made, sensitivities = newton.decompose()(newton(y, x0, x0, 1.0))

    combined = made[0].abs() * sensitivities[0].abs().square().sum(0).sqrt().reshape(N, N)
    truth = image.abs()
    ours = _relative(combined, truth)
    theirs = _relative(bt.nlinv(kspace.reshape(COILS, 1, N, N), i=8).abs().reshape(N, N), truth)
    assert ours < 0.2
    # The same model, the same number of steps: the two answers are the same
    # reconstruction, not merely both plausible.
    assert abs(ours - theirs) < 0.02


def _relative(made: torch.Tensor, truth: torch.Tensor) -> float:
    """The error after the one scale a Gauss-Newton fit leaves free."""
    scale = (made * truth).sum() / (made * made).sum()
    return float((scale * made - truth).norm() / truth.norm())


def test_more_steps_fit_better(problem):
    image, _, kspace = problem
    truth = image.abs()
    errors = []
    for steps in (2, 5, 9):
        newton = nlop.GaussNewton((COILS, N, N), iterations=steps)
        y = newton.prepare()(kspace, _ones())
        x0 = newton.start()
        made, sens = newton.decompose()(newton(y, x0, x0, 1.0))
        errors.append(
            _relative(made[0].abs() * sens[0].abs().square().sum(0).sqrt().reshape(N, N), truth)
        )
    assert errors[0] > errors[1] > errors[2]


def test_the_steps_compose_the_way_the_unrolled_form_does(problem):
    """Two cells at ``alpha`` and ``alpha / redu`` are the two-step operator.

    Which is what says the weight schedule inside BART's unrolled form is the
    one written here -- ``(alpha - alpha_min) / redu + alpha_min``.
    """
    _, _, kspace = problem
    both = nlop.GaussNewton((COILS, N, N), iterations=2, redu=3.0)
    y = both.prepare()(kspace, _ones())
    x0 = both.start()
    together = both(y, x0, x0, 1.0)

    apart = x0
    for alpha in (1.0, 1.0 / 3.0):
        cell = nlop.GaussNewton((COILS, N, N), iterations=1)
        apart = cell(cell.prepare()(kspace, _ones()), apart, x0, alpha)
    torch.testing.assert_close(together, apart, rtol=1e-4, atol=1e-5)


# --- and it differentiates -------------------------------------------------------


#: A coil weighting float32 can hold the gradient of.  BART's default is
#: ``b = 32``, which is ``(1 + 220 |k|^2)^-16``: measured over the state of a
#: 16 by 16 fit, that puts a few per cent of the gradient below float32's
#: smallest normal number and a little under half of the weight's gradient at
#: exactly zero.  What happens past that edge is the platform's business --
#: a norm that is no longer a normal number is one BART's `checkeps` declines
#: to iterate on, and the solve comes back untouched -- so the tests that
#: measure a gradient measure it where the arithmetic has room.  The docstring
#: says the same thing to a caller.
_HOLDS = (220.0, 8.0)


@pytest.mark.parametrize("at", ["data", "iterate", "centre", "weight"])
def test_every_argument_carries_a_gradient(problem, at):
    _, _, kspace = problem
    newton = nlop.GaussNewton((COILS, N, N), iterations=2, sobolev=_HOLDS)
    y = newton.prepare()(kspace, _ones())
    x0 = newton.start()

    order = {"data": 0, "iterate": 1, "centre": 2, "weight": 3}[at]
    xs = [y, x0, x0, newton.weight(1.0)]
    tracked = xs[order].clone().requires_grad_(True)
    xs[order] = tracked

    newton(*xs).abs().square().sum().backward()
    assert tracked.grad is not None
    assert torch.isfinite(tracked.grad).all()
    assert torch.any(tracked.grad != 0)


def test_a_gentler_weighting_keeps_the_whole_gradient_in_range(problem):
    """No part of it is a subnormal, which is the claim :data:`_HOLDS` rests on.

    One direction only.  That the default *does* run past the edge is a fact
    about float32 on a particular machine and not something to assert; that
    ``b = 8`` does not is the same on every machine, and it is what the tests
    above stand on.
    """
    _, _, kspace = problem
    newton = nlop.GaussNewton((COILS, N, N), iterations=2, sobolev=_HOLDS)
    y = newton.prepare()(kspace, _ones())
    x0 = newton.start()
    tracked = x0.clone().requires_grad_(True)
    newton(y, tracked, x0, newton.weight(1.0)).abs().square().sum().backward()

    size = tracked.grad.abs()
    assert torch.all(size > torch.finfo(torch.float32).tiny)


def test_a_denoiser_between_two_cells_trains(problem):
    """NLINV-Net's shape, and the gradient finite differences measure."""
    _, _, kspace = problem
    cells = [nlop.GaussNewton((COILS, N, N), iterations=1, sobolev=_HOLDS) for _ in range(2)]

    def run(weight):
        x = cells[0].start()
        alpha = 1.0
        for cell in cells:
            x = cell(cell.prepare()(kspace, _ones()), x, x, alpha)
            x = weight * x
            alpha = alpha / 2.0
        return x.abs().square().sum()

    weight = torch.nn.Parameter(torch.tensor(0.9))
    run(weight).backward()

    h = 1e-3
    measured = (run(torch.tensor(0.9 + h)) - run(torch.tensor(0.9 - h))).item() / (2 * h)
    assert abs(weight.grad.item() - measured) <= 1e-3 * abs(measured)


# --- what it refuses --------------------------------------------------------------


def test_no_steps_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one step"):
        nlop.GaussNewton((COILS, 8, 8), iterations=0)


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError, match="batch is at least one"):
        nlop.GaussNewton((COILS, 8, 8), batch=0)


# --- an unrolled network as one BART operator -----------------------------------


def _cells(n: int = 2):
    return [nlop.GaussNewton((COILS, N, N), iterations=1, sobolev=_HOLDS) for _ in range(n)]


def test_two_cells_chain_into_one_operator(problem):
    """The step's state is held at rank two and a Python operator at DIMS, so
    this is also what the rank bridging in :func:`bartorch.nlop.chain` is for.
    """
    _, _, kspace = problem
    first, second = _cells()
    ya, yb = first.prepare()(kspace, _ones()), second.prepare()(kspace, _ones())
    x0 = first.start()

    whole = nlop.chain(first, second, output=0, input=1)
    made = whole(yb, x0, second.weight(0.5), ya, x0, x0, first.weight(1.0))
    torch.testing.assert_close(made, second(yb, first(ya, x0, x0, 1.0), x0, 0.5), rtol=0, atol=0)


def test_a_torch_denoiser_goes_between_them_inside_the_one_operator(problem):
    """NLINV-Net with the prior in Python and everything else in C.

    What comes out is a single ``nlop``: BART drives the whole unrolled
    network and crosses into Python once a step, for the denoiser alone.
    """
    _, _, kspace = problem
    first, second = _cells()
    ya, yb = first.prepare()(kspace, _ones()), second.prepare()(kspace, _ones())
    x0 = first.start()
    state = first.state_shape

    crossings = []

    def denoise(x):
        crossings.append(1)
        return 0.9 * x

    prior = nlop.FromTorch(denoise, state, state)
    whole = nlop.chain(nlop.chain(first, prior, output=0, input=0), second, output=0, input=1)

    crossings.clear()
    made = whole(yb, x0, second.weight(0.5), ya, x0, x0, first.weight(1.0))
    assert 1 == len(crossings), "the prior should be reached once per application"
    torch.testing.assert_close(
        made, second(yb, 0.9 * first(ya, x0, x0, 1.0), x0, 0.5), rtol=0, atol=0
    )


@pytest.mark.parametrize("at", [0, 1, 2, 3])
def test_the_composed_network_differentiates_by_its_own_arguments(problem, at):
    _, _, kspace = problem
    first, second = _cells()
    ya, yb = first.prepare()(kspace, _ones()), second.prepare()(kspace, _ones())
    x0 = first.start()
    state = first.state_shape

    prior = nlop.FromTorch(lambda x: 0.9 * x, state, state)
    whole = nlop.chain(nlop.chain(first, prior, output=0, input=0), second, output=0, input=1)

    xs = [yb, x0, second.weight(0.5), ya, x0, x0, first.weight(1.0)]
    tracked = xs[at].clone().requires_grad_(True)
    xs[at] = tracked
    whole(*xs).abs().square().sum().backward()
    assert tracked.grad is not None
    assert torch.isfinite(tracked.grad).all()
    assert torch.any(tracked.grad != 0)


def test_a_weight_the_prior_closed_over_does_not_train_through_the_one_operator(problem):
    """Worth knowing before building a network this way.

    Composed into one ``nlop``, the network is BART's to apply, and
    :class:`~bartorch.nlop.FromTorch` answers for the derivative by its
    *input* -- ``torch.func``'s jvp and vjp of the function it was given.  A
    weight the function closed over is not an argument of anything BART knows
    about, so no gradient reaches it.  Keeping the loop in Python is what
    trains a denoiser; composing is what makes the network one operator.
    """
    _, _, kspace = problem
    first, second = _cells()
    ya, yb = first.prepare()(kspace, _ones()), second.prepare()(kspace, _ones())
    x0 = first.start()
    state = first.state_shape

    weight = torch.nn.Parameter(torch.tensor(0.9))
    prior = nlop.FromTorch(lambda x: weight * x, state, state)
    whole = nlop.chain(nlop.chain(first, prior, output=0, input=0), second, output=0, input=1)
    made = whole(yb, x0, second.weight(0.5), ya, x0, x0, first.weight(1.0))
    assert made.grad_fn is None

    # The same network written as a loop does train it.
    loop = second(yb, weight * first(ya, x0, x0, 1.0), x0, 0.5)
    loop.abs().square().sum().backward()
    assert weight.grad is not None and 0.0 != weight.grad
