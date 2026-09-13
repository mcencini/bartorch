"""The nonlinear MRI encodings: the image and the coils fitted together.

BART's ``noir`` model is what ``nlinv`` inverts, so the test that matters is
that a fit driven from here and one driven by ``nlinv`` are the same
arithmetic.  On a grid they are, to the last bit.  Off it the transform
underneath is FINUFFT rather than BART's own gridding, so the agreement is the
same ``1e-3`` the NUFFT operator itself agrees to.
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


def test_a_noncartesian_fit_agrees_with_the_nlinv_tool():
    # Off the grid the transform is FINUFFT rather than BART's own gridding,
    # so this is the tolerance the NUFFT operator itself holds to, not the
    # exactness the Cartesian model reaches.
    n, coils, spokes = 24, 4, 32
    traj = bt.traj(x=n, y=spokes)
    coil_images = (_phantom(n, n) * _coils(coils, n, n)).reshape(coils, 1, n, n)
    A = linop.NUFFT(traj, (coils, 1, n, n), (coils, spokes, n, 1))
    kspace = A(coil_images)
    kspace = kspace * (100.0 / kspace.norm())

    reference = bt.nlinv(kspace, maxiter=4, traj=traj, normalize=False, w=1.0, x=(n, n, 1))
    F = nlop.NoncartesianSense(traj, (coils, n, n))
    fitted = _fit(F, kspace, 4)
    torch.testing.assert_close(
        fitted.reshape(-1) / reference.abs().max(),
        reference.reshape(-1) / reference.abs().max(),
        rtol=2e-3,
        atol=2e-3,
    )


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
