"""The nonlinear MRI encodings: the image and the coils fitted together.

BART's ``noir`` model is what ``nlinv`` inverts, so the test that matters is
that a fit driven from here and one driven by ``nlinv`` are the same
arithmetic.  On a grid they are, to the last bit.

Off the grid they are not, and comparing them there is not a test.  The
transform underneath is FINUFFT rather than BART's own gridding, and two
Gauss-Newton runs that start together drift apart as the steps accumulate: on
one machine they agree to 7e-4 after two steps and to 6e-3 after twelve, and
on another the same comparison fails at four.  What is being measured there is
how far two numerical paths have diverged, which is not a property of the
model.

So off the grid what is held is what does not depend on a path: that the
model's output is the composition it claims to be, and that a fit converges
towards the image it was made from.  How *far* twelve steps get is a path too
-- the inner conjugate gradients stops on a relative residual, so the last
bits decide how many iterations a step takes -- so what is asserted is the
direction and a bound loose enough for the slowest machine seen.
"""

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import linop, nlop, optim


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _coils(coils: int, y: int, x: int) -> torch.Tensor:
    """Smooth sensitivities, one bump per corner."""
    yy, xx = torch.meshgrid(torch.arange(y), torch.arange(x), indexing="ij")
    corners = [(0, 0), (x - 1, 0), (0, y - 1), (x - 1, y - 1)]
    return torch.stack(
        [
            torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.75 * x) ** 2))
            for cx, cy in (corners * coils)[:coils]
        ]
    ).to(torch.complex64)


def _phantom(y: int, x: int) -> torch.Tensor:
    img = torch.zeros(1, y, x, dtype=torch.complex64)
    img[0, y // 4 : 3 * y // 4, x // 4 : 3 * x // 4] = 1.0
    return img


def _start(F) -> torch.Tensor:
    """``nlinv``'s own initialisation: an image of ones and no coils at all."""
    import math

    sizes = [math.prod(shape) for shape in F.ishapes]
    return torch.cat(
        [
            torch.ones(sizes[0], dtype=torch.complex64),
            torch.zeros(sizes[1], dtype=torch.complex64),
        ]
    )


def _fit(F, kspace, steps: int) -> torch.Tensor:
    """The image, fitted the way ``noir2_recon`` fits it."""
    flat = F.flatten(inputs_only=True)
    solution = optim.IRGNM(
        iterations=steps, alpha=1.0, redu=2.0, alpha_min=0.0, cg_maxiter=100, cg_tol=0.1
    )(F.prepare(kspace), flat, x0=_start(F))
    return flat.split(solution)[0]


# --- shape ------------------------------------------------------------------


def test_the_model_takes_an_image_and_a_set_of_coil_coefficients():
    F = nlop.CartesianSense((4, 16, 16))
    assert F.ishapes == ((1, 1, 16, 16), (4, 1, 16, 16))
    assert F.oshapes == ((4, 1, 16, 16),)


def test_the_third_spatial_axis_is_written_out_so_the_coils_land_on_barts_coil_axis():
    # (coils, y, x) is three axes, and BART reads a third axis as a slice.
    # Without the expansion the coils would be a z stack and the Sobolev
    # weighting would smooth across them.
    F = nlop.CartesianSense((4, 16, 16))
    assert 4 == F.ishapes[1][0]
    assert (1, 16, 16) == F.ishapes[1][1:]


def test_a_three_dimensional_problem_is_taken_as_it_stands():
    F = nlop.CartesianSense((4, 8, 16, 16))
    assert F.ishapes == ((1, 8, 16, 16), (4, 8, 16, 16))


def test_the_model_is_asymmetric_off_the_grid_and_returns_coil_images():
    traj = bt.traj(x=16, y=21)
    F = nlop.NoncartesianSense(traj, (4, 16, 16))
    assert F.oshapes == ((4, 1, 16, 16),)
    assert F.kspace_shape == (4, 21, 16, 1)


def test_on_the_grid_the_model_returns_kspace():
    F = nlop.CartesianSense((4, 16, 16))
    assert F.oshapes[0] == F.kspace_shape


# --- the parts it is made of -------------------------------------------------


def test_the_coil_operator_maps_coefficients_to_sensitivities():
    F = nlop.CartesianSense((4, 16, 16))
    C = F.coils
    assert isinstance(C, linop.LinearOperator)
    assert C.ishape == F.ishapes[1]
    assert C(torch.ones(F.ishapes[1], dtype=torch.complex64)).shape == C.oshape


def test_the_coil_weighting_damps_high_spatial_frequencies():
    # The Sobolev weight is (1 + a|k|^2)^(-b/2) with BART's a = 220, b = 32:
    # steep enough that a coefficient far from the centre contributes nothing.
    F = nlop.CartesianSense((1, 16, 16))
    C = F.coils
    centre = torch.zeros(F.ishapes[1], dtype=torch.complex64)
    centre[0, 0, 8, 8] = 1.0
    edge = torch.zeros(F.ishapes[1], dtype=torch.complex64)
    edge[0, 0, 4, 4] = 1.0
    assert C(edge).abs().max() < 1e-6 * C(centre).abs().max()


def test_the_data_operator_grids_a_measurement_into_what_the_model_returns():
    traj = bt.traj(x=16, y=21)
    F = nlop.NoncartesianSense(traj, (4, 16, 16))
    kspace = _rand(*F.kspace_shape)
    assert tuple(F.prepare(kspace).shape) == F.oshapes[0]


def test_on_the_grid_preparing_a_measurement_leaves_it_alone():
    F = nlop.CartesianSense((4, 16, 16))
    kspace = _rand(*F.kspace_shape)
    torch.testing.assert_close(F.prepare(kspace), kspace, rtol=0, atol=0)


# --- the model's own arithmetic ----------------------------------------------


def test_the_model_multiplies_the_image_by_the_weighted_coils():
    F = nlop.CartesianSense((4, 8, 8))
    image = _rand(*F.ishapes[0])
    coefficients = _rand(*F.ishapes[1])
    made = F(image, coefficients)
    coils = F.coils(coefficients)
    expected = F.transform(F.image(image) * coils)
    torch.testing.assert_close(made, expected, rtol=1e-4, atol=1e-5)


def test_the_derivative_by_the_image_is_the_encoding_with_the_coils_in_it():
    F = nlop.CartesianSense((4, 8, 8))
    image = _rand(*F.ishapes[0])
    coefficients = _rand(*F.ishapes[1])
    F.forward(image, coefficients)
    step = _rand(*F.ishapes[0])
    coils = F.coils(coefficients)
    torch.testing.assert_close(
        F.jacobian(0, 0)(step),
        F.transform(F.image(step) * coils),
        rtol=1e-4,
        atol=1e-5,
    )


def test_both_jacobians_satisfy_the_adjoint_identity():
    F = nlop.CartesianSense((4, 8, 8))
    F.forward(_rand(*F.ishapes[0]), _rand(*F.ishapes[1]))
    for at in (0, 1):
        J = F.jacobian(0, at)
        u, v = _rand(*J.ishape), _rand(*J.oshape)
        left = torch.vdot(J(u).flatten(), v.flatten()).real.item()
        right = torch.vdot(u.flatten(), J.adjoint(v).flatten()).real.item()
        assert left == pytest.approx(right, rel=1e-3)


# --- flattening ---------------------------------------------------------------


def test_flattening_lays_the_unknowns_out_one_after_the_other():
    F = nlop.CartesianSense((4, 8, 8))
    flat = F.flatten(inputs_only=True)
    assert flat.ishapes == ((1 * 8 * 8 + 4 * 8 * 8,),)
    assert flat.oshapes == F.oshapes


def test_splitting_gives_back_a_tensor_per_unknown():
    F = nlop.CartesianSense((4, 8, 8))
    flat = F.flatten(inputs_only=True)
    image, coefficients = flat.split(torch.arange(flat.ishapes[0][0]).to(torch.complex64))
    assert tuple(image.shape) == F.ishapes[0]
    assert tuple(coefficients.shape) == F.ishapes[1]
    assert 0 == image.flatten()[0].real


def test_a_flattened_model_computes_what_the_model_computes():
    F = nlop.CartesianSense((4, 8, 8))
    flat = F.flatten(inputs_only=True)
    image, coefficients = _rand(*F.ishapes[0]), _rand(*F.ishapes[1])
    joined = torch.cat([image.reshape(-1), coefficients.reshape(-1)])
    torch.testing.assert_close(flat(joined), F(image, coefficients), rtol=1e-5, atol=1e-6)


def test_a_flattened_model_reaches_a_solver():
    # BART declares its flattened vector at rank one and everything else in
    # the wrapper at sixteen, and `lsqr2_create` checks the rank it was given
    # against the operator's own -- so a flattened model handed to a solver
    # asserted inside BART, which takes the process rather than raising.  The
    # vector is restated at the wrapper's rank; this is what says so.
    F = nlop.CartesianSense((2, 8, 8))
    flat = F.flatten(inputs_only=True)
    flat.forward(torch.zeros(flat.ishape, dtype=torch.complex64))
    J = flat.jacobian()
    assert torch.isfinite(optim.CG(maxiter=3)(_rand(*J.oshape), J)).all()


def test_flattening_the_outputs_too_gives_one_vector_each_way():
    F = nlop.CartesianSense((4, 8, 8))
    flat = F.flatten()
    assert 1 == len(flat.ishapes) == len(flat.oshapes)
    assert flat.oshapes == ((4 * 8 * 8,),)


# --- against nlinv -------------------------------------------------------------


def _cartesian_data(coils: int = 4, n: int = 24):
    coil_maps = _coils(coils, n, n)
    image = _phantom(n, n)
    coil_images = (image * coil_maps).reshape(coils, 1, n, n)
    kspace = (
        torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(coil_images, dim=(-2, -1))), dim=(-2, -1)
        )
        / n
    )
    return kspace * (100.0 / kspace.norm())


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_a_cartesian_fit_is_the_nlinv_tool_to_the_last_bit(steps):
    # `nlinv -w 1` turns off its automatic scaling, and normalize=False leaves
    # the image as the model fitted it; what is left is the same Gauss-Newton
    # over the same model, and it agrees exactly.
    kspace = _cartesian_data()
    reference = bt.nlinv(kspace, maxiter=steps, normalize=False, w=1.0)
    F = nlop.CartesianSense((4, 24, 24))
    fitted = _fit(F, kspace, steps)
    torch.testing.assert_close(fitted.reshape(-1), reference.reshape(-1), rtol=0.0, atol=0.0)


def test_the_noncartesian_model_is_the_composition_it_claims_to_be():
    # Off the grid the model is asymmetric: it returns the normal equations'
    # gridded coil images rather than samples.  This is that plumbing, with no
    # iteration in it and so nothing for two numerical paths to drift apart
    # over -- a model that returned k-space instead would be wrong by orders
    # of magnitude, not by the last bits.
    #
    # The product goes through BART's own tenmul rather than through torch's
    # `*`.  The two agree to the bit on x86 and need not on every platform --
    # one is a scalar loop and the other whatever SIMD torch has -- and that
    # difference has nothing to do with what is being checked here.
    n, coils, spokes = 16, 4, 21
    F = nlop.NoncartesianSense(bt.traj(x=n, y=spokes), (coils, n, n))
    image = _rand(*F.ishapes[0])
    coefficients = _rand(*F.ishapes[1])
    product = nlop.Multiply(F.ishapes[0], tuple(F.coils.oshape))
    torch.testing.assert_close(
        F(image, coefficients),
        F.transform.normal(product(F.image(image), F.coils(coefficients))),
        rtol=1e-5,
        atol=1e-6,
    )


def test_a_noncartesian_fit_converges_to_the_image_it_was_made_from():
    # A property of the answer rather than of the path taken to it.
    #
    # How far it gets is a path: the inner conjugate gradients stops on a
    # relative residual, so a difference in the last bits changes how many
    # iterations a Newton step takes and therefore where twelve steps land.
    # One machine reaches 0.004 and another 0.062 from the same code.  What
    # does not depend on that is the direction -- the error falls at every
    # step, and ends far below where it started.
    n, coils, spokes = 24, 4, 32
    traj = bt.traj(x=n, y=spokes)
    truth = (_phantom(n, n) * _coils(coils, n, n)).reshape(coils, 1, n, n)
    A = linop.NUFFT(traj, (coils, 1, n, n), (coils, spokes, n, 1))
    kspace = A(truth)
    kspace = kspace * (100.0 / kspace.norm())

    errors = []
    for steps in (4, 8, 12):
        F = nlop.NoncartesianSense(traj, (coils, n, n))
        flat = F.flatten(inputs_only=True)
        solution = optim.IRGNM(iterations=steps, alpha=1.0, redu=2.0, cg_maxiter=100, cg_tol=0.1)(
            F.prepare(kspace), flat, x0=_start(F)
        )
        image, coefficients = flat.split(solution)
        made = image.reshape(1, 1, n, n) * F.coils(coefficients)
        # The model fixes the image and the coils only up to a scalar between
        # them, so the product is what there is to compare.
        scaled = truth * (made.norm() / truth.norm())
        errors.append(((made - scaled).norm() / scaled.norm()).item())

    assert errors[0] > errors[1] > errors[2], f"the fit is not converging: {errors}"
    assert errors[-1] < errors[0] / 2, f"twelve steps barely moved: {errors}"
    # Loose enough for the slowest platform seen (0.062) and far below a fit
    # that is not working at all, which starts around 0.5 and stays there.
    assert errors[-1] < 0.15, f"twelve steps left {errors[-1]:.3f} of error"


def test_a_cartesian_fit_recovers_the_image_it_was_made_from():
    n, coils = 24, 4
    coil_maps = _coils(coils, n, n)
    image = _phantom(n, n)
    kspace = _cartesian_data(coils, n)
    F = nlop.CartesianSense((coils, n, n))
    flat = F.flatten(inputs_only=True)
    solution = optim.IRGNM(iterations=10, alpha=1.0, redu=2.0, cg_maxiter=100, cg_tol=0.1)(
        F.prepare(kspace), flat, x0=_start(F)
    )
    fitted, coefficients = flat.split(solution)
    made = fitted.reshape(1, 1, n, n) * F.coils(coefficients)
    truth = (image * coil_maps).reshape(coils, 1, n, n)
    truth = truth * (made.norm() / truth.norm())
    assert (made - truth).norm() / truth.norm() < 0.15


# --- the general recipe --------------------------------------------------------


def test_an_encoding_with_unknown_coils_is_the_product_in_front_of_it():
    shape = (4, 16, 16)
    E = linop.FFT(shape, axes=(-1, -2))
    F = nlop.CoilSense(E)
    assert F.ishapes == ((1, 16, 16), (4, 16, 16))
    assert F.oshapes == (shape,)
    image, coils = _rand(1, 16, 16), _rand(*shape)
    torch.testing.assert_close(F(image, coils), E(image * coils), rtol=1e-4, atol=1e-4)


def test_a_wave_encoding_takes_unknown_coils_the_same_way():
    # Nothing about CoilSense is particular to a transform; anything that maps
    # coil images to data will do, which is the point of it.
    shape = (4, 16, 16)
    pattern = torch.ones(1, 16, 16, dtype=torch.complex64)
    E = linop.Sampling(pattern, shape) @ linop.FFT(shape, axes=(-1, -2))
    F = nlop.CoilSense(E)
    image, coils = _rand(1, 16, 16), _rand(*shape)
    torch.testing.assert_close(F(image, coils), E(image * coils), rtol=1e-4, atol=1e-4)


def test_the_recipe_reaches_gauss_newton_like_any_other_model():
    shape = (2, 12, 12)
    E = linop.FFT(shape, axes=(-1, -2))
    F = nlop.CoilSense(E)
    image, coils = _rand(1, 12, 12), _rand(*shape)
    data = F(image, coils)
    flat = F.flatten(inputs_only=True)
    truth = torch.cat([image.reshape(-1), coils.reshape(-1)])
    # Started at the answer and regularised towards it, the steps have nowhere
    # to go: what this holds is that the model, its derivative and its adjoint
    # all reach the solver and agree with one another.
    solution = optim.IRGNM(iterations=6, alpha=0.01)(data, flat, x0=truth.clone(), xref=truth)
    torch.testing.assert_close(flat(solution), data, rtol=1e-3, atol=1e-4)
    assert (solution - truth).norm() < 1e-3 * truth.norm()


# --- what the tool wrapper says it does ----------------------------------------


def test_the_nlinv_wrapper_normalizes_unless_told_not_to():
    # BART spells this backwards -- `-N` means "do not normalize" -- and the
    # wrapper used to pass the flag when asked to normalize, which did the
    # opposite of what it said.
    kspace = _cartesian_data(4, 16)
    default = bt.nlinv(kspace, maxiter=3)
    asked = bt.nlinv(kspace, maxiter=3, normalize=True)
    refused = bt.nlinv(kspace, maxiter=3, normalize=False)
    torch.testing.assert_close(default, asked, rtol=0, atol=0)
    assert not torch.equal(default, refused)


def test_the_library_reports_the_model_it_built():
    F = nlop.CartesianSense((4, 8, 8))
    assert "cartesian" in repr(F)
    assert "noncartesian" in repr(nlop.NoncartesianSense(bt.traj(x=8, y=11), (4, 8, 8)))
    assert bartorch is not None
