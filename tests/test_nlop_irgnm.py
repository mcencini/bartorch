"""Gauss-Newton with the inner problem anywhere.

BART has the method in two forms.  ``irgnm`` solves the linearized problem
with its own conjugate gradients; ``irgnm2`` hands that problem to a generic
regularized least-squares solver, which is how a regularized ``nlinv`` or
``moba`` works.  Both are one step of :class:`~bartorch.nlop.IRGNMBlock`
looped in Python, so the inner problem can go to any of the solvers in
:mod:`bartorch.optim`; what holds that honest is that each form reproduces
BART's own loop of that form to the bit.
"""

import pytest
import torch

from bartorch import linop, nlop, optim, priors
from bartorch.nlop.base import _chain


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _decay(n: int = 24):
    """``exp(-t p)`` as one BART operator, and data from a known ``p``."""
    t = torch.linspace(0.2, 3.0, n, dtype=torch.complex64)
    F = _chain(nlop.Multiply((n,), (n,)).partial(0, -t), nlop.Exp((n,)))
    truth = torch.full((n,), 0.7, dtype=torch.complex64)
    return F, torch.exp(-t * truth), truth


_SETTINGS = dict(iterations=8, alpha=1.0, redu=2.0, cg_maxiter=30, cg_tol=0.0)


# --- the loop is BART's ---------------------------------------------------


def test_the_python_loop_with_conjugate_gradients_is_the_library_to_the_last_bit():
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    library = nlop.IRGNM(**_SETTINGS)._in_library(data, F, x0=start.clone())
    written_out = nlop.IRGNM(**_SETTINGS, inner=optim.CG())(data, F, x0=start.clone())
    torch.testing.assert_close(written_out, library, rtol=0.0, atol=0.0)


def test_the_loop_without_an_inner_solver_is_the_first_form_to_the_last_bit():
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    library = nlop.IRGNM(**_SETTINGS)._first_in_library(data, F, x0=start.clone())
    looped = nlop.IRGNM(**_SETTINGS)(data, F, x0=start.clone())
    torch.testing.assert_close(looped, library, rtol=0.0, atol=0.0)


def test_with_a_centre_the_first_form_is_noirs_step_and_not_irgnms_arithmetic():
    # italgos.c adds alpha xref and then subtracts alpha x; noir's step
    # (model_net.c), which the block is, subtracts alpha (x - xref).  Without a
    # centre the two are one subtraction.
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    centre = torch.full((24,), 0.3, dtype=torch.complex64)
    library = nlop.IRGNM(**_SETTINGS)._first_in_library(data, F, x0=start.clone(), xref=centre)
    looped = nlop.IRGNM(**_SETTINGS)(data, F, x0=start.clone(), xref=centre)
    assert not torch.equal(looped, library)
    torch.testing.assert_close(looped, library, rtol=1e-6, atol=1e-7)


def test_it_is_the_library_to_the_last_bit_with_a_regularization_centre_too():
    # xref moves twice per step in irgnm2 -- out before the solve and back
    # after it -- which is the part of the loop easiest to get wrong.
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    centre = torch.full((24,), 0.3, dtype=torch.complex64)
    library = nlop.IRGNM(**_SETTINGS)._in_library(data, F, x0=start.clone(), xref=centre)
    written_out = nlop.IRGNM(**_SETTINGS, inner=optim.CG())(data, F, x0=start.clone(), xref=centre)
    torch.testing.assert_close(written_out, library, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("steps", [1, 2, 5, 12])
def test_it_stays_the_library_however_many_steps_are_taken(steps):
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    settings = {**_SETTINGS, "iterations": steps}
    library = nlop.IRGNM(**settings)._in_library(data, F, x0=start.clone())
    written_out = nlop.IRGNM(**settings, inner=optim.CG())(data, F, x0=start.clone())
    torch.testing.assert_close(written_out, library, rtol=0.0, atol=0.0)


def test_the_weight_floor_is_honoured():
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    settings = dict(iterations=6, alpha=1.0, alpha_min=0.0, alpha_min0=0.25, redu=2.0)
    library = nlop.IRGNM(**settings)._in_library(data, F, x0=start.clone())
    written_out = nlop.IRGNM(**settings, inner=optim.CG())(data, F, x0=start.clone())
    torch.testing.assert_close(written_out, library, rtol=0.0, atol=0.0)


def test_the_two_forms_are_the_same_method_and_not_the_same_arithmetic():
    # irgnm carries alpha (xref - x) into the right-hand side and solves for a
    # step; irgnm2 shifts by xref and solves for the iterate.  Equal in exact
    # arithmetic, and not in floating point -- which is worth knowing before
    # someone compares two runs and reports a bug.
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    first = nlop.IRGNM(**_SETTINGS)(data, F, x0=start.clone())
    second = nlop.IRGNM(**_SETTINGS)._in_library(data, F, x0=start.clone())
    assert not torch.equal(first, second)
    torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-5)


# --- the inner problem goes anywhere ---------------------------------------


@pytest.mark.parametrize("build", [optim.CG, optim.ADMM], ids=["cg", "admm"])
def test_an_unregularized_inner_solver_reaches_the_same_answer(build):
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    reference = nlop.IRGNM(**_SETTINGS, inner=optim.CG())(data, F, x0=start.clone())
    made = nlop.IRGNM(**_SETTINGS, inner=build())(data, F, x0=start.clone())
    torch.testing.assert_close(made, reference, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("solver", ["FISTA", "PRIDU", "ADMM"])
def test_a_proximal_inner_solver_reaches_the_same_answer_at_no_threshold(solver):
    # With nothing to threshold, every one of them solves the same quadratic
    # the conjugate gradients does -- which is the check that the term is
    # being applied to the right problem and the weight to the right place.
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    reference = nlop.IRGNM(**_SETTINGS, inner=optim.CG())(data, F, x0=start.clone())
    inner = getattr(optim, solver)(priors.L1(0.0), maxiter=80)
    made = nlop.IRGNM(**_SETTINGS, inner=inner)(data, F, x0=start.clone())
    torch.testing.assert_close(made, reference, rtol=2e-2, atol=2e-3)


def test_a_threshold_pulls_the_answer_away_from_the_unregularized_one():
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    plain = nlop.IRGNM(**_SETTINGS, inner=optim.CG())(data, F, x0=start.clone())
    apart = None
    for weight in (1e-5, 1e-3, 1e-1):
        made = nlop.IRGNM(**_SETTINGS, inner=optim.FISTA(priors.L1(weight), maxiter=60))(
            data, F, x0=start.clone()
        )
        distance = (made - plain).abs().max().item()
        if apart is not None:
            assert distance > apart
        apart = distance


def test_the_inner_step_is_scaled_by_a_power_iteration_as_barts_own_is():
    # moba/iter_l1.c's inverse_fista computes alpha + power(20, normal) and
    # scales by it every step; a fixed step of 0.95 against a normal operator
    # whose largest eigenvalue is near five diverges outright.
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    made = nlop.IRGNM(**_SETTINGS, inner=optim.FISTA(priors.L1(0.0), maxiter=40, eigen=False))(
        data, F, x0=start.clone()
    )
    assert torch.isfinite(made).all()


def test_a_solver_handed_in_is_not_changed_by_the_run():
    inner = optim.FISTA(priors.L1(0.001), maxiter=20, eigen=False, cclambda=0.0)
    F, data, _ = _decay()
    nlop.IRGNM(**_SETTINGS, inner=inner)(data, F, x0=torch.full((24,), 0.1, dtype=torch.complex64))
    assert inner.eigen is False
    assert 0.0 == inner.cclambda


# --- what a caller may hand in ---------------------------------------------


def test_a_solver_named_rather_than_built_says_to_build_it():
    """``inner`` takes the handle, so that the solver's terms and settings are
    visible where the solve is written rather than defaulted out of sight."""
    with pytest.raises(TypeError, match="not the name of one"):
        nlop.IRGNM(inner="cg")
    with pytest.raises(TypeError, match="optim.FISTA"):
        nlop.IRGNM(inner="fista")


def test_something_that_is_not_a_solver_at_all_says_so():
    with pytest.raises(TypeError, match="bartorch.optim"):
        nlop.IRGNM(inner=object())


def test_no_inner_solver_runs_the_first_form():
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    plain = nlop.IRGNM(**_SETTINGS)
    assert plain.inner is None
    assert torch.isfinite(plain(data, F, x0=start)).all()


def test_the_repr_says_which_inner_solver_is_in_use():
    assert "inner" not in repr(nlop.IRGNM())
    assert "CG" in repr(nlop.IRGNM(inner=optim.CG()))


# --- it fits something ------------------------------------------------------


@pytest.mark.parametrize("build", [None, optim.CG, optim.ADMM], ids=["library", "cg", "admm"])
def test_the_fit_recovers_the_parameter_it_was_made_from(build):
    F, data, truth = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    inner = None if build is None else build()
    made = nlop.IRGNM(iterations=14, alpha=1.0, redu=2.0, cg_maxiter=50, inner=inner)(
        data, F, x0=start
    )
    torch.testing.assert_close(made.real, truth.real, rtol=1e-2, atol=1e-2)


def test_a_linear_operator_is_taken_as_a_nonlinear_one():
    A = linop.FFT((4, 4), axes=-1)
    x = _rand(4, 4)
    made = nlop.IRGNM(iterations=8, alpha=0.001, inner=optim.CG())(
        A(x), A, x0=torch.zeros(4, 4, dtype=torch.complex64)
    )
    torch.testing.assert_close(made, x, rtol=1e-2, atol=1e-2)


def test_a_two_unknown_model_reaches_it_through_flatten():
    shape = (2, 8, 8)
    F = nlop.CartesianSense(shape)
    flat = F._flatten(inputs_only=True)
    import math

    sizes = [math.prod(s) for s in F.ishapes]
    start = torch.cat(
        [
            torch.ones(sizes[0], dtype=torch.complex64),
            torch.zeros(sizes[1], dtype=torch.complex64),
        ]
    )
    kspace = _rand(*F.kspace_shape)
    kspace = kspace * (100.0 / kspace.norm())
    made = nlop.IRGNM(iterations=4, alpha=1.0, inner=optim.CG())(F.prepare(kspace), flat, x0=start)
    assert torch.isfinite(made).all()
    image, coefficients = flat.split(made)
    assert tuple(image.shape) == F.ishapes[0]
    assert tuple(coefficients.shape) == F.ishapes[1]


def test_the_function_is_its_class():
    A = linop.FFT((4, 4), axes=-1)
    x = _rand(4, 4)
    start = torch.zeros(4, 4, dtype=torch.complex64)
    torch.testing.assert_close(
        nlop.irgnm(A(x), A, x0=start.clone(), inner=optim.CG(), iterations=6, alpha=0.01),
        nlop.IRGNM(iterations=6, alpha=0.01, inner=optim.CG())(A(x), A, x0=start.clone()),
        rtol=0.0,
        atol=0.0,
    )


# --- it differentiates --------------------------------------------------------


class _Shrink(torch.nn.Module):
    def forward(self, x, sigma=None):
        return 0.9 * x


def test_a_proximal_inner_solver_carries_the_gradient_finite_differences_measure(monkeypatch):
    """Through the linearization at every step, and through the inner solve's
    implicit backward pass; exact inner solves, so the measurement is the method's."""
    monkeypatch.setattr(optim.ADMMBlock, "_cg_eps", 0.0)
    torch.manual_seed(0)
    F, data, _ = _decay()
    start = torch.full((24,), 0.1, dtype=torch.complex64)
    direction = _rand(24)
    inner = optim.ADMM(priors.ImplicitPrior(_Shrink()), maxiter=4, rho=0.5, cg_maxiter=60)
    schedule = nlop.IRGNM(iterations=3, alpha=1.0, redu=2.0, inner=inner)

    def loss(x0):
        return schedule(data, F, x0=x0).abs().square().sum()

    tracked = start.clone().requires_grad_(True)
    loss(tracked).backward()
    along = torch.real(torch.sum(tracked.grad.conj() * direction)).item()

    h = 3e-3
    with torch.no_grad():
        measured = (loss(start + h * direction) - loss(start - h * direction)).item() / (2 * h)
    assert abs(along - measured) <= 5e-3 * abs(measured)
