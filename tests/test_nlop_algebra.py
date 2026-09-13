"""The nonlinear operator algebra: many arguments, and the combinators that build them.

BART's ``nlop_s`` maps many inputs to many outputs, and ``nlops/chain.h`` is
what puts several of them together.  These tests hold the Python bookkeeping
-- which argument survives a combinator, and in what order -- against what
BART actually built, and check the arithmetic of every basic operator against
torch.
"""

import pytest
import torch

from bartorch import linop, nlop, optim


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _inner(a, b):
    return torch.vdot(a.flatten(), b.flatten()).real.item()


# --- arity ------------------------------------------------------------------


def test_a_two_input_operator_reports_both_of_its_domains():
    M = nlop.Multiply((1, 4, 4), (3, 4, 4))
    assert M.ishapes == ((1, 4, 4), (3, 4, 4))
    assert M.oshapes == ((3, 4, 4),)
    assert M.oshape == (3, 4, 4)


def test_the_single_shape_of_a_many_input_operator_is_refused_by_name():
    M = nlop.Multiply((4,), (4,))
    with pytest.raises(ValueError, match="has 2 inputs"):
        M.ishape


def test_a_one_to_one_operator_still_answers_for_its_single_shape():
    E = nlop.Exp((4, 4))
    assert E.ishape == (4, 4) and E.oshape == (4, 4)
    assert E.ishapes == ((4, 4),) and E.oshapes == ((4, 4),)


def test_the_recorded_shapes_are_held_against_what_bart_built():
    # Every combinator works out its own shapes; _check_shapes is what keeps
    # that bookkeeping honest, so it has to actually look at BART.
    made = nlop.combine(nlop.Exp((4,)), nlop.Log((2, 3)))
    assert made.ishapes == ((4,), (2, 3))
    assert made.oshapes == ((4,), (2, 3))


# --- the tensor product -----------------------------------------------------


def test_the_tensor_product_is_the_pointwise_product_of_its_two_inputs():
    a, b = _rand(1, 4, 4), _rand(3, 4, 4)
    M = nlop.Multiply((1, 4, 4), (3, 4, 4))
    torch.testing.assert_close(M(a, b), a * b, rtol=1e-5, atol=1e-6)


def test_the_derivative_of_a_product_by_one_input_is_multiplication_by_the_other():
    a, b = (
        _rand(
            4,
        ),
        _rand(
            4,
        ),
    )
    M = nlop.Multiply((4,), (4,))
    M.forward(a, b)
    d = _rand(4)
    torch.testing.assert_close(M.jacobian(0, 0)(d), d * b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(M.jacobian(0, 1)(d), a * d, rtol=1e-5, atol=1e-6)


def test_a_jacobian_is_a_linear_operator_with_the_adjoint_identity():
    M = nlop.Multiply((4,), (4,))
    a, b = _rand(4), _rand(4)
    M.forward(a, b)
    J = M.jacobian(0, 1)
    assert isinstance(J, linop.LinearOperator)
    u, v = _rand(4), _rand(4)
    assert _inner(J(u), v) == pytest.approx(_inner(u, J.adjoint(v)), rel=1e-4)


def test_a_jacobian_follows_the_point_the_operator_was_last_applied_at():
    M = nlop.Multiply((4,), (4,))
    J = M.jacobian(0, 0)
    d = torch.ones(4, dtype=torch.complex64)
    b = _rand(4)
    M.forward(_rand(4), b)
    torch.testing.assert_close(J(d), b, rtol=1e-5, atol=1e-6)
    c = _rand(4)
    M.forward(_rand(4), c)
    torch.testing.assert_close(J(d), c, rtol=1e-5, atol=1e-6)


def test_a_python_defined_operator_has_no_jacobian_to_take():
    class F(nlop.NonlinearOperator):
        ishapes = oshapes = ((4,),)

        def forward(self, x):
            return x * x

        def derivative(self, dx):
            return dx

        def adjoint(self, dy):
            return dy

    with pytest.raises(NotImplementedError, match="linearize"):
        F().jacobian()


# --- combine ----------------------------------------------------------------


def test_combine_puts_two_operators_side_by_side_sharing_nothing():
    C = nlop.combine(nlop.Exp((4,)), nlop.Log((3,)))
    assert C.ishapes == ((4,), (3,))
    assert C.oshapes == ((4,), (3,))
    x, y = _rand(4), _rand(3)
    a, b = C(x, y)
    torch.testing.assert_close(a, torch.exp(x), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(b, torch.log(y), rtol=1e-4, atol=1e-5)


def test_an_operator_cannot_be_combined_with_itself():
    E = nlop.Exp((4,))
    with pytest.raises(ValueError, match="cannot be put together with itself"):
        nlop.combine(E, E)


# --- chain ------------------------------------------------------------------


def test_chain_feeds_one_output_into_one_input():
    x = _rand(4)
    c = nlop.chain(nlop.Exp((4,)), nlop.Log((4,)))
    torch.testing.assert_close(c(x), torch.log(torch.exp(x)), rtol=1e-4, atol=1e-4)


def test_chain_keeps_the_arguments_neither_side_used():
    # exp(x), then multiplied by coils: the product's other input survives.
    model = nlop.chain(nlop.Exp((4,)), nlop.Multiply((4,), (4,)), output=0, input=0)
    assert model.ishapes == ((4,), (4,))
    assert model.oshapes == ((4,),)
    x, coils = _rand(4), _rand(4)
    torch.testing.assert_close(model(coils, x), torch.exp(x) * coils, rtol=1e-4, atol=1e-4)


def test_a_chain_refuses_two_arguments_that_do_not_have_the_same_shape():
    with pytest.raises(ValueError, match="a chain needs them to agree"):
        nlop.chain(nlop.Exp((4,)), nlop.Log((3,)))


# --- link and BART's evaluation order ---------------------------------------


def test_a_link_reads_the_output_that_was_produced_first():
    # combine applies its second operand first, so the producer goes second.
    x = _rand(4)
    made = nlop.combine(nlop.Log((4,)), nlop.Exp((4,))).link(1, 0)
    assert made.ishapes == ((4,),) and made.oshapes == ((4,),)
    torch.testing.assert_close(made(x), torch.log(torch.exp(x)), rtol=1e-4, atol=1e-4)


def test_a_link_the_other_way_round_is_refused_and_not_silently_wrong():
    # BART's operator_combi_create applies its operands in reverse, and
    # nothing in BART complains if a link reads a buffer nothing wrote; the
    # answer is simply garbage.  This is the one thing the stage bookkeeping
    # exists for.
    made = nlop.combine(nlop.Exp((4,)), nlop.Log((4,)))
    with pytest.raises(ValueError, match="produced after"):
        made.link(0, 1)


def test_chain_is_a_combine_and_a_link_in_the_order_that_works():
    x = _rand(4)
    by_hand = nlop.combine(nlop.Log((4,)), nlop.Exp((4,))).link(1, 0)
    by_chain = nlop.chain(nlop.Exp((4,)), nlop.Log((4,)))
    torch.testing.assert_close(by_hand(x), by_chain(x), rtol=1e-5, atol=1e-6)


def test_a_link_refuses_two_arguments_of_different_shapes():
    made = nlop.combine(nlop.Log((3,)), nlop.Exp((4,)))
    with pytest.raises(ValueError, match="a link needs them to agree"):
        made.link(1, 0)


# --- dup, stack, permute, del_out, pin ---------------------------------------


def test_dup_makes_two_inputs_one():
    D = nlop.Multiply((4,), (4,)).dup(0, 1)
    assert D.ishapes == ((4,),)
    x = _rand(4)
    torch.testing.assert_close(D(x), x * x, rtol=1e-5, atol=1e-6)


def test_the_derivative_after_a_dup_is_the_sum_of_both():
    # d(x*x) = 2 x dx, which is only right if both derivatives were added.
    D = nlop.Multiply((4,), (4,)).dup(0, 1)
    x = _rand(4)
    D.forward(x)
    d = _rand(4)
    torch.testing.assert_close(D.jacobian(0, 0)(d), 2 * x * d, rtol=1e-4, atol=1e-5)


def test_dup_takes_its_inputs_in_order():
    M = nlop.Multiply((4,), (4,))
    with pytest.raises(ValueError, match="in order"):
        M.dup(1, 0)


def test_dup_refuses_inputs_of_different_shapes():
    made = nlop.combine(nlop.Exp((4,)), nlop.Log((3,)))
    with pytest.raises(ValueError, match="only inputs of one shape"):
        made.dup(0, 1)


def test_stacking_two_outputs_concatenates_them_along_an_axis():
    S = nlop.combine(nlop.Exp((2, 4)), nlop.Log((2, 4))).stack_outputs(0, 1, 0)
    assert S.oshapes == ((4, 4),)
    assert S.ishapes == ((2, 4), (2, 4))
    x, y = _rand(2, 4), _rand(2, 4)
    torch.testing.assert_close(
        S(x, y), torch.cat([torch.exp(x), torch.log(y)]), rtol=1e-4, atol=1e-4
    )


def test_stacking_two_inputs_splits_one_tensor_between_them():
    S = nlop.Multiply((2, 4), (2, 4)).stack_inputs(0, 1, 0)
    assert S.ishapes == ((4, 4),)
    x = _rand(4, 4)
    torch.testing.assert_close(S(x), x[:2] * x[2:], rtol=1e-5, atol=1e-6)


def test_stacking_refuses_arguments_that_differ_away_from_the_axis():
    made = nlop.combine(nlop.Exp((2, 4)), nlop.Log((2, 3)))
    with pytest.raises(ValueError, match="differ away from axis"):
        made.stack_outputs(0, 1, 0)


def test_permuting_inputs_reorders_what_the_operator_takes():
    P = nlop.combine(nlop.Exp((4,)), nlop.Log((3,))).permute_inputs([1, 0])
    assert P.ishapes == ((3,), (4,))
    assert P.oshapes == ((4,), (3,))
    x, y = _rand(4), _rand(3)
    a, b = P(y, x)
    torch.testing.assert_close(a, torch.exp(x), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b, torch.log(y), rtol=1e-4, atol=1e-4)


def test_permuting_outputs_reorders_what_the_operator_returns():
    P = nlop.combine(nlop.Exp((4,)), nlop.Log((3,))).permute_outputs([1, 0])
    assert P.oshapes == ((3,), (4,))
    x, y = _rand(4), _rand(3)
    a, b = P(x, y)
    torch.testing.assert_close(a, torch.log(y), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b, torch.exp(x), rtol=1e-4, atol=1e-4)


def test_a_permutation_has_to_take_every_argument_once():
    made = nlop.combine(nlop.Exp((4,)), nlop.Log((3,)))
    with pytest.raises(ValueError, match="takes each of them once"):
        made.permute_inputs([0, 0])


def test_shifting_an_argument_moves_it_and_closes_the_gap():
    made = nlop.combine(nlop.Exp((4,)), nlop.combine(nlop.Log((3,)), nlop.Sqrt((2,))))
    assert made.ishapes == ((4,), (3,), (2,))
    assert made.shift_input(0, 2).ishapes == ((2,), (4,), (3,))


def test_dropping_an_output_leaves_the_rest():
    D = nlop.combine(nlop.Exp((4,)), nlop.Log((3,))).del_out(1)
    assert D.oshapes == ((4,),)
    assert D.ishapes == ((4,), (3,))
    x, y = _rand(4), _rand(3)
    torch.testing.assert_close(D(x, y), torch.exp(x), rtol=1e-4, atol=1e-4)


def test_the_only_output_cannot_be_dropped():
    with pytest.raises(ValueError, match="nothing to compute"):
        nlop.Exp((4,)).del_out(0)


def test_pinning_an_input_fixes_it_to_a_tensor():
    coils = _rand(3, 4, 4)
    P = nlop.Multiply((1, 4, 4), (3, 4, 4)).pin(1, coils)
    assert P.ishapes == ((1, 4, 4),)
    x = _rand(1, 4, 4)
    torch.testing.assert_close(P(x), x * coils, rtol=1e-5, atol=1e-6)


def test_a_pinned_operator_does_not_hold_the_caller_s_tensor():
    # BART copies, so writing to the tensor afterwards must change nothing.
    coils = torch.ones(4, dtype=torch.complex64)
    P = nlop.Multiply((4,), (4,)).pin(1, coils)
    coils.fill_(7.0)
    x = _rand(4)
    torch.testing.assert_close(P(x), x, rtol=1e-5, atol=1e-6)


def test_a_constant_operator_takes_nothing_and_returns_its_tensor():
    value = _rand(4)
    K = nlop.Constant(value)
    assert K.ishapes == ()
    torch.testing.assert_close(K(), value, rtol=1e-5, atol=1e-6)


# --- generic application -----------------------------------------------------


def test_applying_an_operator_with_the_wrong_number_of_inputs_says_so():
    M = nlop.Multiply((4,), (4,))
    with pytest.raises(ValueError, match="takes 2 inputs"):
        M(_rand(4))


def test_every_output_comes_back_as_its_own_tensor():
    C = nlop.combine(nlop.Exp((4,)), nlop.Log((3,)))
    out = C(_rand(4), _rand(3))
    assert isinstance(out, tuple) and 2 == len(out)
    assert (4,) == tuple(out[0].shape) and (3,) == tuple(out[1].shape)


# --- autograd ----------------------------------------------------------------


def test_autograd_through_a_two_input_operator_matches_torch():
    a = _rand(4).requires_grad_(True)
    b = _rand(4).requires_grad_(True)
    M = nlop.Multiply((4,), (4,))
    M(a, b).abs().pow(2).sum().backward()
    x = a.detach().clone().requires_grad_(True)
    y = b.detach().clone().requires_grad_(True)
    (x * y).abs().pow(2).sum().backward()
    torch.testing.assert_close(a.grad, x.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(b.grad, y.grad, rtol=1e-4, atol=1e-5)


def test_a_gradient_is_taken_only_where_one_was_asked_for():
    a = _rand(4).requires_grad_(True)
    b = _rand(4)
    M = nlop.Multiply((4,), (4,))
    M(a, b).abs().pow(2).sum().backward()
    assert a.grad is not None


# --- the basic operators, against torch --------------------------------------


@pytest.mark.parametrize(
    "make, reference",
    [
        (lambda s: nlop.Exp(s), torch.exp),
        (lambda s: nlop.Log(s), torch.log),
        (lambda s: nlop.Sqrt(s), torch.sqrt),
        (lambda s: nlop.Abs(s), lambda x: x.abs().to(torch.complex64)),
        (lambda s: nlop.Phase(s), lambda x: x / x.abs()),
        (lambda s: nlop.Inverse(s), lambda x: 1 / x),
        (lambda s: nlop.Power(s, 2.0), lambda x: x**2),
        (lambda s: nlop.Add(s, 1 + 2j), lambda x: x + (1 + 2j)),
    ],
    ids=["exp", "log", "sqrt", "abs", "phase", "inverse", "power", "add"],
)
def test_an_elementwise_operator_is_what_torch_computes(make, reference):
    x = _rand(3, 4) + 2.0  # away from zero, where several of these are singular
    torch.testing.assert_close(make((3, 4))(x), reference(x), rtol=1e-4, atol=1e-4)


def test_the_smooth_absolute_value_is_differentiable_where_the_sharp_one_is_not():
    S = nlop.SmoothAbs((4,), eps=1e-2)
    x = torch.zeros(4, dtype=torch.complex64)
    S.forward(x)
    d = _rand(4)
    assert torch.isfinite(S.derivative(d)).all()


def test_the_root_sum_of_squares_reduces_the_axes_it_was_given():
    R = nlop.RootSumOfSquares((3, 4), axes=0)
    assert R.oshape == (1, 4)
    x = _rand(3, 4)
    torch.testing.assert_close(
        R(x),
        x.abs().pow(2).sum(0, keepdim=True).sqrt().to(torch.complex64),
        rtol=1e-4,
        atol=1e-4,
    )


def test_the_sum_of_squares_reduces_the_axes_it_was_given():
    S = nlop.Sum((3, 4), axes=0)
    assert S.oshape == (1, 4)
    x = _rand(3, 4)
    torch.testing.assert_close(
        S(x), x.abs().pow(2).sum(0, keepdim=True).to(torch.complex64), rtol=1e-4, atol=1e-4
    )


def test_a_quotient_divides_its_first_input_by_its_second():
    D = nlop.Divide((4,))
    a, b = _rand(4), _rand(4) + 2.0
    torch.testing.assert_close(D(a, b), a / b, rtol=1e-4, atol=1e-4)


def test_a_weighted_sum_scales_each_of_its_two_inputs():
    W = nlop.Weighted((4,), a=2.0, b=-3.0)
    x, y = _rand(4), _rand(4)
    torch.testing.assert_close(W(x, y), 2 * x - 3 * y, rtol=1e-5, atol=1e-5)


# --- the derivative of every basic operator, against a finite difference ------


@pytest.mark.parametrize(
    "make",
    [
        lambda s: nlop.Exp(s),
        lambda s: nlop.Log(s),
        lambda s: nlop.Sqrt(s),
        lambda s: nlop.Inverse(s),
        lambda s: nlop.Power(s, 3.0),
    ],
    ids=["exp", "log", "sqrt", "inverse", "power"],
)
def test_a_derivative_agrees_with_a_finite_difference(make):
    F = make((6,))
    x = _rand(6) + 3.0
    d = _rand(6)
    F.forward(x)
    predicted = F.derivative(d)
    h = 1e-4
    taken = (F.forward(x + h * d) - F.forward(x - h * d)) / (2 * h)
    torch.testing.assert_close(predicted, taken, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize(
    "make",
    [
        lambda s: nlop.Exp(s),
        lambda s: nlop.Sqrt(s),
        lambda s: nlop.Inverse(s),
    ],
    ids=["exp", "sqrt", "inverse"],
)
def test_the_adjoint_of_a_derivative_is_the_adjoint_of_what_it_applies(make):
    F = make((6,))
    F.forward(_rand(6) + 3.0)
    u, v = _rand(6), _rand(6)
    assert _inner(F.derivative(u), v) == pytest.approx(_inner(u, F.adjoint(v)), rel=1e-3)


# --- the algebra reaching the rest of the library ----------------------------


def test_a_composed_model_is_solved_by_gauss_newton():
    # exp(-t p) fitted by nlinv's own solver, built out of the algebra rather
    # than out of a torch function: a constant, a product and an exponential.
    # Away from t = 0, where the model says nothing about the parameter.
    t = torch.linspace(0.2, 3.0, 16, dtype=torch.complex64)
    truth = torch.full((16,), 0.7, dtype=torch.complex64)
    data = torch.exp(-t * truth)

    F = nlop.chain(nlop.Multiply((16,), (16,)).pin(0, -t), nlop.Exp((16,)))
    assert F.ishapes == ((16,),)
    fitted = optim.IRGNM(iterations=12)(data, F, x0=torch.full((16,), 0.1, dtype=torch.complex64))
    torch.testing.assert_close(fitted.real, truth.real, rtol=1e-2, atol=1e-2)


def test_a_linear_operator_still_composes_with_a_nonlinear_one():
    shape = (4, 4)
    F = linop.FFT(shape, axes=-1)
    E = nlop.Exp(shape)
    made = F @ E
    x = _rand(*shape)
    torch.testing.assert_close(made(x), F(torch.exp(x)), rtol=1e-4, atol=1e-4)
