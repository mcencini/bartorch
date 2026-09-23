"""Each primitive's derivative as an operator of the linearization point.

What a bundle claims is pinned against torch and against finite differences;
the comparison with ``nlop_get_derivative`` is BART against BART and says only
that the two routes have not drifted apart.
"""

import pytest
import torch

from bartorch import linop, nlop

SHAPE = (6,)


def rand(shape, generator):
    real = torch.randn(shape, generator=generator)
    imag = torch.randn(shape, generator=generator)
    return (real + 1j * imag).to(torch.complex64)


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(20260915)


#: One case per declared bundle: the operator, and a torch function that is it.
#: Points are kept away from zero where the derivative has a pole there.
ELEMENTWISE = {
    "exp": (nlop.Exp(SHAPE), torch.exp, 0.0),
    "log": (nlop.Log(SHAPE), torch.log, 3.0),
    "sqrt": (nlop.Sqrt(SHAPE), torch.sqrt, 3.0),
    "inverse": (nlop.Inverse(SHAPE), lambda x: 1.0 / x, 3.0),
    "power": (nlop.Power(SHAPE, 2.0), lambda x: x**2, 3.0),
    "add": (nlop.Add(SHAPE, 1 + 2j), lambda x: x + (1 + 2j), 0.0),
}


def point(name, generator):
    return rand(SHAPE, generator) + ELEMENTWISE[name][2]


# --- what the members are ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_a_member_takes_the_tangent_first_and_the_point_after(name):
    op = ELEMENTWISE[name][0]
    bundle = op._bundled
    assert bundle.derivative.ishapes == (*op.ishapes, *op.ishapes)
    assert bundle.derivative.oshapes == op.oshapes
    assert bundle.adjoint.ishapes == (*op.oshapes, *op.ishapes)
    assert bundle.adjoint.oshapes == op.ishapes


def test_the_tangent_comes_first_for_a_model_of_two_unknowns():
    op = nlop.Multiply((1, 6), (3, 6))
    bundle = op._bundled
    assert bundle.derivative.ishapes == ((1, 6), (3, 6), (1, 6), (3, 6))
    assert bundle.adjoint.ishapes == ((3, 6), (1, 6), (3, 6))
    assert bundle.adjoint.oshapes == ((1, 6), (3, 6))


# --- against something outside BART ------------------------------------------


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_the_derivative_is_torchs_jacobian_vector_product(name, generator):
    op, fn, _ = ELEMENTWISE[name]
    x, dx = point(name, generator), rand(SHAPE, generator)
    want = torch.func.jvp(fn, (x,), (dx,))[1]
    assert torch.allclose(op._bundled.derivative(dx, x), want, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_the_derivative_is_the_limit_of_a_difference_quotient(name, generator):
    op, _, _ = ELEMENTWISE[name]
    x, dx = point(name, generator), rand(SHAPE, generator)

    # A central quotient, whose error is quadratic in the step: at 1e-2 that
    # is a few parts in ten thousand, which single precision still resolves.
    eps = 1e-2
    quotient = (op.forward(x + eps * dx) - op.forward(x - eps * dx)) / (2 * eps)

    exact = op._bundled.derivative(dx, x)
    assert (quotient - exact).abs().max() < 5e-3 * exact.abs().max()


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_the_adjoint_satisfies_the_adjoint_identity(name, generator):
    op, _, _ = ELEMENTWISE[name]
    x, dx, dz = point(name, generator), rand(SHAPE, generator), rand(SHAPE, generator)
    forward = (op._bundled.derivative(dx, x).conj() * dz).sum()
    back = (dx.conj() * op._bundled.adjoint(dz, x)).sum()
    assert abs(forward - back) < 1e-4 * abs(forward)


@pytest.mark.parametrize("name", sorted(set(ELEMENTWISE) - {"add"}))
def test_the_adjoint_is_not_the_transpose(name, generator):
    """The near miss a real-valued test would never see.

    ``add`` is left out because its diagonal is the real number one, where the
    two agree and nothing is being missed.
    """
    op, _, _ = ELEMENTWISE[name]
    x, dz = point(name, generator), rand(SHAPE, generator)
    adjoint = op._bundled.adjoint(dz, x)
    transpose = op._bundled.adjoint(dz.conj(), x).conj()
    assert not torch.allclose(adjoint, transpose, atol=1e-3)


def test_the_product_rule_is_what_multiply_differentiates_by(generator):
    op = nlop.Multiply((1, 6), (3, 6))
    a, b = rand((1, 6), generator), rand((3, 6), generator)
    da, db = rand((1, 6), generator), rand((3, 6), generator)
    assert torch.allclose(op._bundled.derivative(da, db, a, b), da * b + a * db, atol=1e-5)

    dz = rand((3, 6), generator)
    dx_a, dx_b = op._bundled.adjoint(dz, a, b)
    assert torch.allclose(dx_a, (b.conj() * dz).sum(0, keepdim=True), atol=1e-5)
    assert torch.allclose(dx_b, a.conj() * dz, atol=1e-5)


def test_the_normal_is_the_derivative_followed_by_the_adjoint(generator):
    op = nlop.Multiply((1, 6), (3, 6))
    a, b = rand((1, 6), generator), rand((3, 6), generator)
    da, db = rand((1, 6), generator), rand((3, 6), generator)
    got = op._bundled.normal(da, db, a, b)
    want = op._bundled.adjoint(op._bundled.derivative(da, db, a, b), a, b)
    for one, other in zip(got, want):
        assert torch.allclose(one, other, atol=1e-5)


# --- the point as an argument ------------------------------------------------


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_the_point_is_an_argument_and_not_the_last_forward(name, generator):
    """A forward somewhere else between building and applying changes nothing."""
    op, _, _ = ELEMENTWISE[name]
    x, dx = point(name, generator), rand(SHAPE, generator)
    want = op._bundled.derivative(dx, x)
    op.forward(point(name, generator))
    assert torch.equal(op._bundled.derivative(dx, x), want)


def test_a_linear_operators_bundle_ignores_the_point(generator):
    transform = linop.FFT(SHAPE, axes=(-1,))
    bundle = transform.to_nonlinear()._bundled
    dx, dz = rand(SHAPE, generator), rand(SHAPE, generator)
    for x in (rand(SHAPE, generator), rand(SHAPE, generator)):
        assert torch.equal(bundle.derivative(dx, x), transform.forward(dx))
        assert torch.equal(bundle.adjoint(dz, x), transform.adjoint(dz))


def test_a_constant_has_no_cotangent_to_return(generator):
    made = nlop.Constant(rand(SHAPE, generator))
    assert made._bundled.adjoint is None
    assert torch.equal(made._bundled.derivative(), torch.zeros(SHAPE, dtype=torch.complex64))
    with pytest.raises(NotImplementedError, match="no inputs"):
        _ = made._bundled.normal


def test_a_torch_operators_bundle_differentiates_the_function(generator):
    fn = lambda p: p * torch.exp(-p)  # noqa: E731
    made = nlop.TorchOperator(fn, SHAPE, SHAPE)
    x, dx, dz = rand(SHAPE, generator), rand(SHAPE, generator), rand(SHAPE, generator)

    forward = made._bundled.derivative(dx, x)
    assert torch.allclose(forward, torch.func.jvp(fn, (x,), (dx,))[1], atol=1e-5)

    paired = (forward.conj() * dz).sum()
    back = (dx.conj() * made._bundled.adjoint(dz, x)).sum()
    assert abs(paired - back) < 1e-4 * abs(paired)


# --- BART against BART -------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_the_bundle_agrees_with_the_derivative_at_the_stored_point(name, generator):
    """An agreement check between two routes into BART, not a numerical test."""
    op, _, _ = ELEMENTWISE[name]
    x, dx = point(name, generator), rand(SHAPE, generator)
    op.forward(x)
    assert torch.allclose(op._bundled.derivative(dx, x), op._derivative(dx), atol=1e-5, rtol=1e-5)
    dz = rand(SHAPE, generator)
    assert torch.allclose(op._bundled.adjoint(dz, x), op._adjoint(dz), atol=1e-5, rtol=1e-5)


# --- the chain rule ----------------------------------------------------------


def jacobians(op, xs, dxs, dzs):
    """What BART's own derivative at the stored point answers, summed over arguments."""
    op.forward(*xs)
    jac = [[op._jacobian(o, i) for i in range(len(op.ishapes))] for o in range(len(op.oshapes))]
    forward = [
        sum(jac[o][i].forward(dxs[i]) for i in range(len(op.ishapes)))
        for o in range(len(op.oshapes))
    ]
    back = [
        sum(jac[o][i].adjoint(dzs[o]) for o in range(len(op.oshapes)))
        for i in range(len(op.ishapes))
    ]
    return forward, back


def tupled(made):
    return (made,) if isinstance(made, torch.Tensor) else tuple(made)


def composed(generator):
    """One composition per node of the algebra, each with a bundle."""
    from bartorch.nlop.base import _chain, _combine

    exp, log = nlop.Exp(SHAPE), nlop.Log(SHAPE)
    return {
        "chain": nlop.Exp(SHAPE) @ nlop.Log(SHAPE),
        "chain2": _chain(nlop.Exp(SHAPE), nlop.Multiply(SHAPE, SHAPE), output=0, input=1),
        "combine": _combine(exp, log),
        "dup": nlop.Multiply(SHAPE, SHAPE)._dup(0, 1),
        "permute_inputs": nlop.Multiply((1, 6), (3, 6))._permute_inputs([1, 0]),
        "reshape_input": nlop.Exp(SHAPE)._reshape_input(0, (1, 6)),
        "reshape_output": nlop.Exp(SHAPE)._reshape_output(0, (1, 6)),
        "del_out": _combine(nlop.Exp(SHAPE), nlop.Log(SHAPE))._del_out(1),
        "partial": nlop.Multiply(SHAPE, SHAPE).partial(1, rand(SHAPE, generator) + 3.0),
        "nested": _chain(
            nlop.Exp(SHAPE) @ nlop.Log(SHAPE), nlop.Multiply(SHAPE, SHAPE), output=0, input=1
        ),
        # The operand that feeds the chain has an output left over, which is
        # the arithmetic the one-output cases never reach.
        "chain from several outputs": _chain(
            _combine(nlop.Exp(SHAPE), nlop.Log(SHAPE)),
            nlop.Multiply(SHAPE, SHAPE),
            output=0,
            input=1,
        ),
        "chain from the later output": _chain(
            _combine(nlop.Exp(SHAPE), nlop.Log(SHAPE)),
            nlop.Multiply(SHAPE, SHAPE),
            output=1,
            input=0,
        ),
        "chain into several outputs": _chain(
            nlop.Multiply(SHAPE, SHAPE),
            _combine(nlop.Exp(SHAPE), nlop.Log(SHAPE)),
            output=0,
            input=1,
        ),
    }


@pytest.mark.parametrize("name", sorted(composed(torch.Generator().manual_seed(0))))
def test_a_composed_bundle_is_barts_own_derivative(name, generator):
    """An agreement check: the chain rule against what BART differentiates for itself."""
    op = composed(generator)[name]
    xs = [rand(s, generator) + 3.0 for s in op.ishapes]
    dxs = [rand(s, generator) for s in op.ishapes]
    dzs = [rand(s, generator) for s in op.oshapes]
    want_d, want_a = jacobians(op, xs, dxs, dzs)

    for got, want in zip(tupled(op._bundled.derivative(*dxs, *xs)), want_d):
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
    for got, want in zip(tupled(op._bundled.adjoint(*dzs, *xs)), want_a):
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", sorted(composed(torch.Generator().manual_seed(0))))
def test_a_composed_bundle_satisfies_the_adjoint_identity(name, generator):
    op = composed(generator)[name]
    xs = [rand(s, generator) + 3.0 for s in op.ishapes]
    dxs = [rand(s, generator) for s in op.ishapes]
    dzs = [rand(s, generator) for s in op.oshapes]
    forward = tupled(op._bundled.derivative(*dxs, *xs))
    back = tupled(op._bundled.adjoint(*dzs, *xs))
    paired = sum((one.conj() * dz).sum() for one, dz in zip(forward, dzs))
    other = sum((dx.conj() * one).sum() for dx, one in zip(dxs, back))
    assert abs(paired - other) < 1e-4 * abs(paired)


def test_the_chain_rule_is_torchs_own_over_the_written_out_composition(generator):
    from bartorch.nlop.base import _chain

    op = _chain(nlop.Exp(SHAPE), nlop.Multiply(SHAPE, SHAPE), output=0, input=1)
    fn = lambda a, x: a * torch.exp(x)  # noqa: E731

    a, x = rand(SHAPE, generator), rand(SHAPE, generator)
    da, dx = rand(SHAPE, generator), rand(SHAPE, generator)
    want = torch.func.jvp(fn, (a, x), (da, dx))[1]
    assert torch.allclose(op._bundled.derivative(da, dx, a, x), want, atol=1e-5, rtol=1e-4)

    dz = rand(SHAPE, generator)
    back = torch.func.vjp(fn, a, x)[1](dz)
    for got, one in zip(op._bundled.adjoint(dz, a, x), back):
        assert torch.allclose(got, one, atol=1e-5, rtol=1e-4)


def test_a_composed_bundle_does_not_move_with_a_forward_elsewhere(generator):
    op = nlop.Exp(SHAPE) @ nlop.Log(SHAPE)
    x, dx = rand(SHAPE, generator) + 3.0, rand(SHAPE, generator)
    want = op._bundled.derivative(dx, x)
    op.forward(rand(SHAPE, generator) + 3.0)
    assert torch.equal(op._bundled.derivative(dx, x), want)


def test_a_link_has_no_bundle():
    """A tie is a feedback edge, and its rule needs the whole graph rather than the node."""
    from bartorch.nlop.base import _combine

    # BART applies a combination back to front, so `Log` produces before `Exp`
    # consumes and the tie is the one the algebra allows.
    made = _combine(nlop.Exp(SHAPE), nlop.Log(SHAPE))._link(1, 0)
    assert made._bundled is None


@pytest.mark.parametrize("name", sorted(composed(torch.Generator().manual_seed(0))))
def test_a_composed_normal_is_its_own_derivative_followed_by_its_own_adjoint(name, generator):
    """The chain rule puts the first operand in both members, and the normal in one
    graph twice; both copies are handed the same point, so the derivative it stores
    is the same either way."""
    op = composed(generator)[name]
    if 1 != len(op.oshapes):
        pytest.skip("a normal operator is defined for one output")
    xs = [rand(s, generator) + 3.0 for s in op.ishapes]
    dxs = [rand(s, generator) for s in op.ishapes]
    got = tupled(op._bundled.normal(*dxs, *xs))
    want = tupled(op._bundled.adjoint(op._bundled.derivative(*dxs, *xs), *xs))
    for one, other in zip(got, want):
        assert torch.equal(one, other)


# --- the primitives BART builds out of others --------------------------------

#: Each with the composition it declares, and whether it is complex-linear in
#: its tangent.  The ones built on ``zss`` are not: they carry a conjugation.
DERIVED = {
    "divide": (nlop.Divide(SHAPE), None, True),
    "divide (regularised)": (nlop.Divide(SHAPE, 1e-3), None, True),
    "sum": (nlop.SumOfSquares(SHAPE, ()), lambda x: (x.conj() * x).real.to(torch.complex64), False),
    "sum over an axis": (nlop.SumOfSquares(SHAPE, (-1,)), None, False),
    "root sum of squares": (nlop.RootSumOfSquares(SHAPE, ()), lambda x: x.abs() + 0j, False),
    "root sum of squares (regularised)": (nlop.RootSumOfSquares(SHAPE, (-1,), 1e-3), None, False),
    "abs": (nlop.Abs(SHAPE), lambda x: x.abs() + 0j, False),
    "smooth abs": (nlop.SmoothAbs(SHAPE, 1e-6), None, False),
}


@pytest.mark.parametrize("name", sorted(DERIVED))
def test_the_declared_composition_answers_what_bart_answers(name, generator):
    """What BART builds the operator out of, written here, is the operator."""
    op = DERIVED[name][0]
    xs = [rand(shape, generator) + 3.0 for shape in op.ishapes]
    assert torch.equal(op.forward(*xs), op._composition().forward(*xs))


@pytest.mark.parametrize("name", sorted(DERIVED))
def test_a_derived_bundle_is_barts_own_derivative(name, generator):
    """An agreement check: the chain rule over the pieces against `nlop_get_derivative`."""
    op = DERIVED[name][0]
    xs = [rand(shape, generator) + 3.0 for shape in op.ishapes]
    dxs = [rand(shape, generator) for shape in op.ishapes]
    dzs = [rand(shape, generator) for shape in op.oshapes]
    want_d, want_a = jacobians(op, xs, dxs, dzs)

    for got, want in zip(tupled(op._bundled.derivative(*dxs, *xs)), want_d):
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
    for got, want in zip(tupled(op._bundled.adjoint(*dzs, *xs)), want_a):
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", sorted(DERIVED))
def test_a_derived_derivative_is_torchs_jacobian_vector_product(name, generator):
    op, fn, _ = DERIVED[name]
    if fn is None:
        pytest.skip("no torch function writes this one in one line")
    x, dx = rand(SHAPE, generator) + 3.0, rand(SHAPE, generator)
    want = torch.func.jvp(fn, (x,), (dx,))[1]
    assert torch.allclose(op._bundled.derivative(dx, x), want, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", sorted(DERIVED))
def test_a_derived_adjoint_is_adjoint_over_the_reals(name, generator):
    """``zss`` conjugates its own input, so what is linear is real-linear.

    The identity therefore holds in the real inner product and not the complex
    one, which is what ``linop.Real`` already says of itself.  Asserting the
    complex one would be asserting a different operator.
    """
    op, _, complex_linear = DERIVED[name]
    xs = [rand(shape, generator) + 3.0 for shape in op.ishapes]
    dxs = [rand(shape, generator) for shape in op.ishapes]
    dzs = [rand(shape, generator) for shape in op.oshapes]

    forward = sum(
        (one.conj() * dz).sum() for one, dz in zip(tupled(op._bundled.derivative(*dxs, *xs)), dzs)
    )
    back = sum(
        (dx.conj() * one).sum() for dx, one in zip(dxs, tupled(op._bundled.adjoint(*dzs, *xs)))
    )

    assert abs(forward.real - back.real) < 1e-4 * abs(forward)
    if complex_linear:
        assert abs(forward - back) < 1e-4 * abs(forward)
    else:
        assert abs(forward - back) > 1e-3 * abs(forward)


def test_the_phase_operator_follows_from_the_two_it_is_built_of(generator):
    """``zphsr`` is ``zabs`` into ``zdiv`` duplicated, and declares nothing of its own."""
    op = nlop.Phase(SHAPE)
    x, dx, dz = rand(SHAPE, generator) + 3.0, rand(SHAPE, generator), rand(SHAPE, generator)
    want_d, want_a = jacobians(op, [x], [dx], [dz])
    assert torch.allclose(op._bundled.derivative(dx, x), want_d[0], atol=1e-5, rtol=1e-4)
    assert torch.allclose(op._bundled.adjoint(dz, x), want_a[0], atol=1e-5, rtol=1e-4)
