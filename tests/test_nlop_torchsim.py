"""TorchSim signal models as BART nonlinear operators.

The bridge has to hold three things: that the physics agrees with BART's own
(``bart signal`` computes the same closed forms), that the derivative and its
adjoint are what they claim to be, and that what comes out reaches BART's
solvers and the operator algebra like anything else.
"""

import math

import pytest
import torch

import bartorch.tools as bt
from bartorch import linop, nlop, optim

torchsim = pytest.importorskip("torchsim")


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _inner(a, b):
    return torch.vdot(a.flatten(), b.flatten()).real.item()


# --- shape and packing --------------------------------------------------------


def test_a_model_maps_parameter_channels_to_one_image_per_contrast():
    M = nlop.MultiEcho((10.0, 20.0, 40.0, 80.0), (8, 8))
    assert M.ishape == (3, 8, 8)  # T2, and the amplitude's two halves
    assert M.oshape == (4, 8, 8)
    assert M.names == ("T2", "amplitude.real", "amplitude.imag")


def test_turning_the_amplitude_off_drops_its_two_channels():
    M = nlop.MultiEcho((10.0, 20.0), (4, 4), amplitude=False)
    assert M.ishape == (1, 4, 4)
    assert M.names == ("T2",)


def test_solving_for_more_than_one_property_adds_a_channel_each():
    M = nlop.MultiEcho(
        (10.0, 20.0, 40.0),
        (4, 4),
        unknown=("T2", "offset"),
        bounds={"T2": (1.0, 500.0), "offset": (-1.0, 1.0)},
    )
    assert M.ishape == (4, 4, 4)
    assert M.names[:2] == ("T2", "offset")


def test_a_starting_point_and_the_maps_it_stands_for_round_trip():
    M = nlop.MultiEcho((10.0, 20.0), (5, 6))
    start = M.initial(T2=80.0)
    assert tuple(start.shape) == M.ishape
    assert start.dtype == torch.complex64
    maps = M.split(start)
    torch.testing.assert_close(maps["T2"], torch.full((5, 6), 80.0), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(
        maps["amplitude"], torch.ones(5, 6, dtype=torch.complex64), rtol=1e-5, atol=1e-6
    )


def test_a_bound_keeps_every_iterate_physical():
    # The variable solved for is a transform of the property, so no value the
    # solver can reach maps outside the bound.
    M = nlop.MultiEcho((10.0,), (4,), bounds={"T2": (20.0, 100.0)})
    for value in (-50.0, 0.0, 50.0):
        wild = torch.full(M.ishape, value, dtype=torch.complex64)
        T2 = M.split(wild)["T2"]
        assert (20.0 <= T2).all() and (T2 <= 100.0).all()


# --- the physics, against BART's own -------------------------------------------


def test_the_multi_echo_decay_is_what_bart_computes():
    # `signal -S` is BART's spin echo: exp(-n TE / T2), n = 0 .. measurements-1.
    # Two libraries, two implementations, one closed form.
    T2_ms, TE_ms, steps = 50.0, 10.0, 6
    reference = bt.signal(
        S=True, e=TE_ms * 1e-3, n=steps, **{"2": (T2_ms * 1e-3, T2_ms * 1e-3, 1)}
    ).reshape(-1)
    M = nlop.MultiEcho(tuple(TE_ms * k for k in range(steps)), ())
    made = M(M.initial(T2=T2_ms)).reshape(-1)
    torch.testing.assert_close(made.real, reference.real, rtol=1e-5, atol=1e-6)


def test_the_multi_echo_decay_is_the_exponential_it_says_it_is():
    TE = (5.0, 15.0, 35.0)
    M = nlop.MultiEcho(TE, (3, 3))
    made = M(M.initial(T2=40.0))
    expected = torch.stack([torch.full((3, 3), math.exp(-t / 40.0)) for t in TE])
    torch.testing.assert_close(made.real, expected, rtol=1e-4, atol=1e-5)


def test_the_inversion_recovery_is_the_recovery_it_says_it_is():
    TI = (50.0, 400.0, 1100.0, 2500.0)
    M = nlop.InversionRecovery(TI, (3, 3))
    made = M(M.initial(T1=1000.0))
    expected = torch.stack([torch.full((3, 3), 1.0 - 2.0 * math.exp(-t / 1000.0)) for t in TI])
    torch.testing.assert_close(made.real, expected, rtol=1e-4, atol=1e-5)


def test_the_amplitude_scales_the_whole_curve():
    M = nlop.MultiEcho((10.0, 20.0), (2, 2))
    plain = M(M.initial(T2=50.0))
    scaled = M(M.initial(T2=50.0, amplitude=2.0 - 1.0j))
    torch.testing.assert_close(scaled, plain * (2.0 - 1.0j), rtol=1e-4, atol=1e-5)


# --- the derivative and its adjoint ---------------------------------------------


def test_the_derivative_agrees_with_a_finite_difference():
    M = nlop.MultiEcho((5.0, 15.0, 35.0, 70.0), (4,))
    at = M.initial(T2=60.0)
    step = torch.randn(M.ishape, dtype=torch.complex64).real.to(torch.complex64)
    M.forward(at)
    predicted = M.derivative(step)
    h = 1e-4
    taken = (M.forward(at + h * step) - M.forward(at - h * step)) / (2 * h)
    torch.testing.assert_close(predicted, taken, rtol=2e-2, atol=2e-4)


def test_the_adjoint_is_the_adjoint_of_the_derivative():
    M = nlop.MultiEcho((5.0, 15.0, 35.0, 70.0), (4, 4))
    M.forward(M.initial(T2=60.0))
    u = _rand(*M.ishape).real.to(torch.complex64)
    v = _rand(*M.oshape)
    assert _inner(M.derivative(u), v) == pytest.approx(_inner(u, M.adjoint(v)), rel=1e-3)


def test_the_imaginary_half_of_the_domain_is_a_null_direction():
    # TorchSim's maps are real; they are carried in the real part of a complex
    # buffer, and nothing reads the other half.  An iterate that starts real
    # therefore stays real, to the bit.
    M = nlop.MultiEcho((10.0, 20.0), (4, 4))
    at = M.initial(T2=50.0)
    torch.testing.assert_close(M(at + 3.0j * torch.ones_like(at)), M(at), rtol=0, atol=0)
    M.forward(at)
    assert 0.0 == M.derivative(1.0j * torch.ones(M.ishape, dtype=torch.complex64)).abs().max()
    assert 0.0 == M.adjoint(_rand(*M.oshape)).imag.abs().max()


# --- reaching the solvers and the algebra ----------------------------------------


def test_gauss_newton_recovers_the_relaxation_time_it_was_given():
    TE = (5.0, 15.0, 35.0, 70.0, 120.0)
    M = nlop.MultiEcho(TE, (4, 4))
    truth = M.initial(T2=45.0)
    data = M(truth)
    start = M.initial(T2=120.0)
    fitted = optim.IRGNM(iterations=14, alpha=1.0, redu=2.0, cg_maxiter=50)(
        data, M, x0=start.clone(), xref=start
    )
    torch.testing.assert_close(M.split(fitted)["T2"], torch.full((4, 4), 45.0), rtol=5e-2, atol=2.0)


def test_a_model_chains_with_an_encoding_into_one_operator():
    TE = (5.0, 20.0, 60.0)
    shape = (3, 8, 8)
    M = nlop.MultiEcho(TE, (8, 8))
    E = linop.FFT(shape, axes=(-1, -2))
    F = nlop.chain(M, E.to_nonlinear())
    assert F.ishapes == (M.ishape,)
    assert F.oshapes == (shape,)
    maps = M.initial(T2=50.0)
    torch.testing.assert_close(F(maps), E(M(maps)), rtol=1e-4, atol=1e-5)


def test_a_model_under_an_encoding_is_solved_for_its_parameters():
    TE = (5.0, 20.0, 60.0, 120.0)
    shape = (4, 6, 6)
    M = nlop.MultiEcho(TE, (6, 6))
    E = linop.FFT(shape, axes=(-1, -2))
    F = nlop.chain(M, E.to_nonlinear())
    truth = M.initial(T2=40.0)
    data = F(truth)
    start = M.initial(T2=120.0)
    fitted = optim.IRGNM(iterations=14, alpha=1.0, redu=2.0, cg_maxiter=50)(
        data, F, x0=start.clone(), xref=start
    )
    torch.testing.assert_close(M.split(fitted)["T2"], torch.full((6, 6), 40.0), rtol=1e-1, atol=4.0)


def test_a_model_differentiates_in_torch():
    M = nlop.MultiEcho((10.0, 30.0), (3, 3))
    x = M.initial(T2=50.0).requires_grad_(True)
    M(x).abs().pow(2).sum().backward()
    assert x.grad is not None
    assert 0.0 == x.grad.imag.abs().max()


# --- any simulator at all ---------------------------------------------------------


def test_an_arbitrary_simulator_becomes_an_operator():
    from torchsim.simulators import FSESimulator

    M = nlop.Bloch(
        FSESimulator(flip=torch.full((6,), 180.0), ESP=8.0, TR=3000.0),
        "T1",
        "T2",
        shape=(2, 2),
        bounds={"T1": (100.0, 4000.0), "T2": (5.0, 500.0)},
    )
    assert M.names[:2] == ("T1", "T2")
    assert M.ishape == (4, 2, 2)
    made = M(M.initial(T1=1000.0, T2=80.0))
    assert made.shape[1:] == (2, 2)
    assert torch.isfinite(made).all()


def test_an_arbitrary_simulator_has_a_working_adjoint():
    from torchsim.simulators import InversionRecoverySimulator

    M = nlop.Bloch(
        InversionRecoverySimulator(TI=(100.0, 500.0, 2000.0)),
        "T1",
        shape=(3,),
        bounds={"T1": (50.0, 4000.0)},
    )
    M.forward(M.initial(T1=900.0))
    u = _rand(*M.ishape).real.to(torch.complex64)
    v = _rand(*M.oshape)
    assert _inner(M.derivative(u), v) == pytest.approx(_inner(u, M.adjoint(v)), rel=1e-3)


def test_a_model_operator_can_be_handed_over_directly():
    from torchsim.recon import ModelOperator
    from torchsim.simulators import MultiEchoSimulator

    model = ModelOperator(MultiEchoSimulator(TE=(10.0, 40.0)), "T2", bounds={"T2": (1.0, 500.0)})
    M = nlop.FromTorchSim(model, (2, 2))
    assert M.ishape == (3, 2, 2) and M.oshape == (2, 2, 2)
    assert "MultiEchoSimulator" in repr(M)


# --- the dependency ----------------------------------------------------------------


def test_torchsim_is_a_dependency_and_not_an_extra():
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text())
    required = " ".join(data["project"]["dependencies"])
    assert "torchsim" in required
    for name, entries in data["project"].get("optional-dependencies", {}).items():
        assert not any("torchsim" in entry for entry in entries), (
            f"torchsim is in the {name} extra as well as in dependencies"
        )
