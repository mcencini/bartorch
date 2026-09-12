"""The regularization terms, applied on their own.

A term is BART's: ``opt_reg_configure`` turns it into a proximal operator and,
for some of them, a transform to apply it through.  Handing the pair to a
solver is what ``pics`` does, and what :mod:`bartorch.optim` did until now --
which meant a term could not be evaluated at all outside a solve.

It can now, and the claim is that it is the same operator: BART's own IST,
written out here around ``Regularizer.prox``, is the solver's answer to the
bit.
"""

import pytest
import torch

from bartorch import linop, optim, prox
from bartorch._dispatch import BartError


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


SHAPE = (1, 8, 8)
AXES = (-1, -2)


def _terms():
    return {
        "L1": prox.L1(0.1),
        "L2": prox.L2(0.1),
        "Wavelet": prox.Wavelet(AXES, 0.1),
        "LocallyLowRank": prox.LocallyLowRank(AXES, 0.1, block=4),
        "NonNegative": prox.NonNegative(),
        "FourierL1": prox.FourierL1(AXES, 0.1),
        "Laplace": prox.Laplace(AXES, 0.1),
        "ImaginaryL1": prox.ImaginaryL1(0.1),
    }


# --- the same operator the solver uses ---------------------------------------


def test_a_term_applied_here_is_the_one_bart_iterates_with():
    """BART's `ist`, written out: threshold, residual, step, and a last
    threshold because `italgo_config` does not ask for the fast form.

    Not close -- the same bits.  The arithmetic is BART's throughout: the
    proximal operator is the solver's own and so is the normal operator, and
    what is left for torch is two axpys on an image.
    """
    torch.manual_seed(0)
    A = linop.FFT(SHAPE, axes=AXES)
    y = A(_rand(*SHAPE))
    term = prox.L1(0.05)
    step, iters = 0.7, 6

    theirs = optim.IST(term, maxiter=iters, step=step)(y, A)

    b = A.adjoint(y)
    x = torch.zeros_like(b)
    for _ in range(iters):
        x = term.prox(x, step)
        x = x + step * (b - A.normal(x))
    ours = term.prox(x, step)

    assert torch.equal(ours, theirs)


# --- what a term's proximal operator is --------------------------------------


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_the_l1_term_is_a_soft_threshold_at_gamma_times_its_weight(gamma):
    """Which says what ``gamma`` means: the step, with the weight already in
    the operator."""
    torch.manual_seed(0)
    x = _rand(*SHAPE)
    got = prox.L1(0.2).prox(x, gamma)
    want = torch.polar((x.abs() - gamma * 0.2).clamp(min=0), x.angle()).to(torch.complex64)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_the_non_negative_term_clamps_both_parts():
    """Worth writing down: it is not a constraint on the real part alone.  A
    complex image loses its negative imaginary parts to this term too."""
    z = torch.tensor([-1 + 2j, 3 - 1j, 0.5 + 0j, -2 - 3j], dtype=torch.complex64).reshape(1, 1, 4)
    got = prox.NonNegative().prox(z, 1.0)
    want = torch.complex(z.real.clamp(min=0), z.imag.clamp(min=0))
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("name", sorted(_terms()))
def test_every_term_applies_on_the_shape_it_says(name):
    torch.manual_seed(0)
    term = _terms()[name]
    shape = term.prox_shape(SHAPE)
    out = term.prox(_rand(*shape), 0.5, image_shape=SHAPE)
    assert tuple(out.shape) == shape
    assert out.dtype == torch.complex64


@pytest.mark.parametrize("name", sorted(_terms()))
def test_most_terms_work_on_the_image_itself(name):
    """The ones that carry their transform inside the proximal operator, which
    is all of them but total variation and its relatives."""
    assert _terms()[name].prox_shape(SHAPE) == SHAPE


def test_total_variation_thresholds_the_components_of_a_gradient():
    """So its proximal operator is not shaped like an image, and BART's own
    IST -- which applies the operator to the image and ignores the transform --
    cannot take this term."""
    shape = prox.TotalVariation(AXES, 0.1).prox_shape(SHAPE)
    assert shape[0] == 2
    assert [n for n in shape if n != 1] == [2, 8, 8]


def test_a_tensor_that_is_not_what_the_image_makes_is_refused():
    """The image shape is what the term is configured for; the tensor is what
    its proximal operator takes.  For most terms they are the same, so a
    caller need not say -- and when they are not, saying wrong is caught."""
    with pytest.raises(ValueError, match="works on"):
        prox.L1(0.1).prox(_rand(1, 2, 8), 1.0, image_shape=SHAPE)


def test_total_variation_applies_over_the_image_it_was_told_about():
    torch.manual_seed(0)
    term = prox.TotalVariation(AXES, 0.1)
    shape = term.prox_shape(SHAPE)
    out = term.prox(_rand(*shape), 0.5, image_shape=SHAPE)
    assert tuple(out.shape) == shape


# --- the transform in front of it --------------------------------------------


def test_a_wavelet_term_carries_its_transform_inside_the_proximal_operator():
    """Which is why it works on the image's own shape: what is in front is the
    identity."""
    T = prox.Wavelet(AXES, 0.1).transform(SHAPE)
    x = _rand(*SHAPE)
    torch.testing.assert_close(T(x), x, rtol=0, atol=0)


def test_the_laplace_term_carries_a_real_transform_in_front_of_it():
    """And its codomain is shaped like the image, so the shapes do not say
    which arrangement a term is.  A caller that guessed from the shape would
    quietly leave this one out."""
    T = prox.Laplace(AXES, 0.1).transform(SHAPE)
    x = _rand(*SHAPE)
    assert (T(x) - x).abs().max() > 1e-3


def test_a_transform_past_barts_rank_is_refused_rather_than_truncated():
    """`linop_domain` fills what it is given and returns the rank it has; the
    rank is what says a gradient does not fit, not the dimensions."""
    with pytest.raises(BartError, match="rank 17"):
        prox.TotalVariation(AXES, 0.1).transform(SHAPE)


@pytest.mark.parametrize("name", sorted(_terms()))
def test_a_transform_maps_the_image_somewhere_it_can_be_thresholded(name):
    term = _terms()[name]
    T = term.transform(SHAPE)
    assert T.ishape == SHAPE
    assert T.oshape == term.prox_shape(SHAPE)


def test_cycle_spinning_makes_the_wavelet_threshold_a_random_one():
    """`randshift` shifts the transform by a draw of BART's own before every
    application, so the same term applied twice to the same image answers
    twice differently.

    Worth stating, because it is what an iteration written out here has to
    respect: two loops agree to the bit only if they apply this term the same
    number of times, in the same order.  Turned off, it is a function again.
    """
    x = _rand(*SHAPE)
    spun = prox.Wavelet(AXES, 0.1)
    assert not torch.equal(spun.prox(x, 1.0), spun.prox(x, 1.0))

    still = prox.Wavelet(AXES, 0.1, randshift=False)
    assert torch.equal(still.prox(x, 1.0), still.prox(x, 1.0))


def test_a_term_builds_its_operator_once_per_shape():
    term = prox.L1(0.1)
    assert term.build(SHAPE) == term.build(SHAPE)
