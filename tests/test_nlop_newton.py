"""BART's Gauss-Newton step over its noir model, taken by IRGNMBlock.

``noir/model_net.c`` builds one iteration of ``nlinv`` as an ``nlop``, out of
``nlop``s throughout: the forward model, the derivative *as a function of the
linearisation point*, the adjoint, and ``norm_inv``'s implicitly
differentiated inverse of the normal operator.  :class:`~bartorch.nlop.IRGNMBlock`
applies the same three operators, so a step differentiates by the data, the
iterate, the regularisation centre and the weight, second-order terms included
-- which is what ``networks/nlinvnet.c`` trains through.

The thing to know about the model is that it has no sampling pattern until it
is given one, and it is given one as a *side effect* of the gridding:
``noir_adjoint_fft_fun`` calls ``linop_gdiag_set_diag`` on its way past.  A
step applied before that reads a diagonal nothing has written, which is a
segmentation fault; preparing the data is what writes it.
"""

import math

import pytest
import torch

import bartorch.tools as bt
from bartorch import linop, nlop, optim

N, COILS = 16, 2

#: What goes to the model, and what goes to the block.
_MODEL = {"pattern", "trajectory", "sobolev", "weights", "basis", "mask", "real", "sos", "c"}
_BLOCK = {"redu", "alpha_min", "cg_maxiter", "cg_tol", "fuse", "inner"}


class _Newton:
    """A noir model, a block over it, and ``iterations`` steps as one call.

    ``newton(y, xn, x0, alpha)`` takes prepared data, the iterate, the centre
    and the weight -- a number or a vector as long as the state.  The pattern
    is the model's, so a step is taken over an encoding that already carries it.
    """

    def __init__(self, image_shape, iterations: int = 1, **settings):
        model = {k: settings.pop(k) for k in list(settings) if k in _MODEL}
        block = {k: settings.pop(k) for k in list(settings) if k in _BLOCK}
        assert not settings, settings
        if model.get("trajectory") is None:
            model.setdefault("pattern", _ones(image_shape[-1]))
        self.model = nlop.NonlinearSense(image_shape, **model)
        self.iterations = iterations
        self.block = nlop.IRGNMBlock(**block)
        self.space = self.block._space(self.model)

    def __getattr__(self, name):
        return getattr(self.space, name)

    @property
    def plan(self):
        return self.block.plan(self.model)

    def weight(self, alpha: float) -> torch.Tensor:
        return torch.full(self.space.state_shape, float(alpha), dtype=torch.complex64)

    def __call__(self, y, xn, x0, alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = self.weight(alpha)
        state = self.block.State(xn, x0, y, alpha, self.space)
        for _ in range(self.iterations):
            state = self.block(state, self.model)
        return state.x


def _cell(image_shape, **settings):
    return _Newton(image_shape, **settings)


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


def _decompose(newton, x):
    """A state through the model's own linear parts: an image and coil profiles.

    What ``noir``'s ``decompose`` does -- the coils go through the Sobolev
    weighting, so what comes back is profiles rather than the coefficients
    that were fitted.
    """
    image, coefficients = newton.split(x)
    model = newton.model
    return model.image(image), model.coils(coefficients)


# --- what it takes and returns ------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
def test_a_batch_is_a_leading_axis_on_the_data(batch):
    newton = _cell((COILS, 8, 8))
    kspace = torch.randn(batch, *newton.model.oshapes[0], dtype=torch.complex64)
    state = newton.block.start(kspace, newton.model)
    assert (batch, *newton.data_shape) == tuple(state.data.shape)
    assert (batch, *newton.state_shape) == tuple(state.x.shape)


def test_the_unknowns_are_the_models_own():
    newton = _cell((COILS, 8, 8))
    model = newton.model
    assert (model._image_only(), model.coefficient_shape) == newton.lowered.ishapes
    assert math.prod(newton.state_shape) == sum(
        math.prod(shape) for shape in newton.lowered.ishapes
    )


def test_off_the_grid_the_step_is_taken_over_the_asymmetric_stage():
    trajectory = bt.traj(x=16, y=8, r=True)
    newton = _cell((COILS, 8, 8), trajectory=trajectory)
    assert newton.plan.fused
    assert "normal" == newton.plan.domain


# --- the iterate ---------------------------------------------------------------


def test_the_start_is_an_image_of_ones_and_no_coils():
    newton = _cell((COILS, 8, 8))
    image, coils = newton.split(newton.start())
    torch.testing.assert_close(image, torch.ones_like(image), rtol=0, atol=0)
    torch.testing.assert_close(coils, torch.zeros_like(coils), rtol=0, atol=0)


def test_split_and_join_are_each_other():
    newton = _cell((COILS, 8, 8))
    x = torch.randn(newton.state_shape, dtype=torch.complex64)
    image, coils = newton.split(x)
    torch.testing.assert_close(newton.join(image, coils), x, rtol=0, atol=0)


def test_decompose_carries_the_transforms_and_split_does_not():
    # `decompose` runs the coils through the Sobolev weighting, so what comes
    # back is profiles rather than the coefficients that were fitted.
    newton = _cell((COILS, 8, 8))
    x = torch.randn(newton.state_shape, dtype=torch.complex64)
    assert not torch.equal(_decompose(newton, x)[1], newton.split(x)[1])


# --- it reconstructs ------------------------------------------------------------


def test_it_recovers_the_image_as_well_as_the_tool_does(problem):
    image, _, kspace = problem
    newton = _cell((COILS, N, N), iterations=8, redu=2.0)
    y = newton.prepare(kspace)
    x0 = newton.start()
    made, sensitivities = _decompose(newton, newton(y, x0, x0, 1.0))

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
        newton = _cell((COILS, N, N), iterations=steps)
        y = newton.prepare(kspace)
        x0 = newton.start()
        made, sens = _decompose(newton, newton(y, x0, x0, 1.0))
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
    both = _cell((COILS, N, N), iterations=2, redu=3.0)
    y = both.prepare(kspace)
    x0 = both.start()
    together = both(y, x0, x0, 1.0)

    apart = x0
    for alpha in (1.0, 1.0 / 3.0):
        cell = _cell((COILS, N, N), iterations=1)
        apart = cell(cell.prepare(kspace), apart, x0, alpha)
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
    newton = _cell((COILS, N, N), iterations=2, sobolev=_HOLDS)
    y = newton.prepare(kspace)
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
    newton = _cell((COILS, N, N), iterations=2, sobolev=_HOLDS)
    y = newton.prepare(kspace)
    x0 = newton.start()
    tracked = x0.clone().requires_grad_(True)
    newton(y, tracked, x0, newton.weight(1.0)).abs().square().sum().backward()

    size = tracked.grad.abs()
    assert torch.all(size > torch.finfo(torch.float32).tiny)


def test_a_denoiser_between_two_cells_trains(problem):
    """NLINV-Net's shape, and the gradient finite differences measure."""
    _, _, kspace = problem
    cells = [_cell((COILS, N, N), iterations=1, sobolev=_HOLDS) for _ in range(2)]

    def run(weight):
        x = cells[0].start()
        alpha = 1.0
        for cell in cells:
            x = cell(cell.prepare(kspace), x, x, alpha)
            x = weight * x
            alpha = alpha / 2.0
        return x.abs().square().sum()

    weight = torch.nn.Parameter(torch.tensor(0.9))
    run(weight).backward()

    h = 1e-3
    measured = (run(torch.tensor(0.9 + h)) - run(torch.tensor(0.9 - h))).item() / (2 * h)
    assert abs(weight.grad.item() - measured) <= 1e-3 * abs(measured)


# --- what it takes and what it refuses -------------------------------------------


def test_a_model_of_the_algebra_steps_over_its_own_bundle():
    model = nlop.CoilSense(linop.FFT((COILS, 1, 8, 8), axes=(-1, -2)))
    assert "chain rule" == nlop.IRGNMBlock().plan(model).bundle


def test_a_model_with_no_bundle_is_refused():
    """A tie has no chain rule of its own, so the composition carrying one has no step."""
    made = nlop.combine(nlop.Exp((4,)), nlop.Log((4,))).link(1, 0)
    with pytest.raises(TypeError, match="no derivative as a function of the point"):
        nlop.IRGNMBlock().start(torch.zeros(4, dtype=torch.complex64), made)


def test_a_model_of_the_algebra_takes_a_batch_too():
    """The batch is the block's, not ``noir2_net``'s, so every model has one."""
    model = nlop.CoilSense(linop.FFT((COILS, 1, 8, 8), axes=(-1, -2)))
    kspace = torch.randn(2, COILS, 1, 8, 8, dtype=torch.complex64)
    assert 2 == nlop.IRGNMBlock().start(kspace, model).x.shape[0]


def test_the_coil_configurations_barts_network_model_refused_are_taken():
    """``noir2_net`` fits coils on the image's grid; the step here does not care."""
    trajectory = bt.traj(x=16, y=8, r=True)
    for extra in ({}, {"oversampling_coils": 1.5}, {"oversampled_coils": True}):
        model = nlop.NoncartesianSense(trajectory, (COILS, 8, 8), **extra)
        assert nlop.IRGNMBlock().plan(model).fused


def test_an_inner_solver_takes_the_linearized_problem(problem):
    """The second form over BART's own model, differentiable by the iterate."""
    _, _, kspace = problem
    newton = _cell((COILS, N, N), iterations=2, sobolev=_HOLDS, inner=optim.CG(maxiter=20))
    y = newton.prepare(kspace)
    x0 = newton.start()
    tracked = x0.clone().requires_grad_(True)
    state = newton.block.State(tracked, x0, y, 1.0, newton.space)
    for _ in range(2):
        state = newton.block(state, newton.model)
    state.x.abs().square().sum().backward()
    assert torch.isfinite(tracked.grad).all()
    assert torch.any(tracked.grad != 0)


# --- a real denoiser, trained between the steps ----------------------------------


class _Denoiser(torch.nn.Module):
    """Two convolutions over the real and imaginary parts, which is what a
    denoiser is shaped like even when it is this small."""

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.view_as_real(x).permute(0, 3, 1, 2)
        made = self.body(parts).permute(0, 2, 3, 1).contiguous()
        return torch.view_as_complex(made)


def test_a_modules_parameters_pack_and_come_back():
    net = _Denoiser()
    weights = nlop.Parameters(net)
    assert (sum(p.numel() for p in net.parameters()),) == weights.shape

    packed = weights.pack()
    assert torch.complex64 == packed.dtype
    made = weights.unpack(packed)
    assert all(torch.equal(made[name], p) for name, p in net.named_parameters())

    weights.load(torch.zeros_like(packed))
    assert all(torch.all(0 == p) for p in net.parameters())


def test_a_vector_of_the_wrong_length_says_so():
    weights = nlop.Parameters(_Denoiser())
    with pytest.raises(ValueError, match="parameters, not"):
        weights.unpack(torch.zeros(3, dtype=torch.complex64))


def test_a_module_with_nothing_to_train_says_so():
    with pytest.raises(ValueError, match="no parameters to train"):
        nlop.Parameters(torch.nn.ReLU())


def test_a_convolutional_denoiser_trains_between_the_steps(problem):
    """NLINV-Net's shape: two steps, a torch module between them, and an
    optimizer over the module's weights that reduces the loss.

    Six updates is not a reconstruction -- what it says is that the gradient is
    a real one and points the way it should.
    """
    image, _, kspace = problem
    torch.manual_seed(0)
    first, second = (_cell((COILS, N, N), iterations=1, sobolev=_HOLDS) for _ in range(2))
    ya, yb = first.prepare(kspace), second.prepare(kspace)
    x0 = first.start()
    pixels = N * N

    net = _Denoiser()

    def prior(x):
        # The denoiser touches the image half of the state; the coil
        # coefficients go through untouched.
        made = net(x[:pixels].reshape(1, N, N))
        return torch.cat([made.reshape(pixels), x[pixels:]])

    optimiser = torch.optim.Adam(net.parameters(), lr=1e-2)
    truth = image.abs().reshape(pixels)

    losses = []
    for _ in range(6):
        optimiser.zero_grad()
        made = second(yb, prior(first(ya, x0, x0, 1.0)), x0, 0.5)[:pixels].abs()
        scale = (made * truth).sum() / (made * made).sum().clamp_min(1e-12)
        loss = ((scale * made - truth) ** 2).sum()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters())
        losses.append(loss.item())
        optimiser.step()

    assert losses[-1] < losses[0]
    assert losses == sorted(losses, reverse=True), "the loss should fall at every step"
