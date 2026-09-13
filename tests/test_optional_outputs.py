"""Commands whose output BART marks optional, and the two that write it anyway.

A derived wrapper asks for as many output arrays as BART requires, and nine
commands require none.  For seven of them that is right: they print their
answer and write a file only when given a name, through ``anon_cfl`` or behind
an ``if``.  Two create it unguarded -- ``mobafit.c:398`` and ``morphop.c:77``
-- and ``create_cfl`` hands the name to ``io_unlink_if_opened``, which calls
``strcmp`` on it.  With no name that is a segmentation fault, which takes the
process rather than raising.
"""

import pytest
import torch

import bartorch.tools as bt
from bartorch._call import _WRITES_ITS_OPTIONAL_OUTPUT, _outputs
from bartorch._catalogue import COMMANDS


def _echoes(n: int = 8, echoes: int = 7):
    """Multi-echo images on BART's echo axis, with the echo times beside them."""
    times = torch.tensor([1.6 * k for k in range(1, echoes + 1)], dtype=torch.complex64)
    return times.reshape(echoes, 1, 1, 1, 1, 1), torch.randn(
        echoes, 1, 1, 1, n, n, dtype=torch.complex64
    )


def test_a_command_that_writes_its_optional_output_is_asked_for_one():
    for name in _WRITES_ITS_OPTIONAL_OUTPUT:
        command = COMMANDS[name]
        assert not any(a.required for a in command.outputs), (
            f"{name} now requires an output, so it no longer needs the exception"
        )
        assert 1 == _outputs(command)


def test_every_other_command_is_asked_for_what_bart_requires():
    for name, command in COMMANDS.items():
        if name in _WRITES_ITS_OPTIONAL_OUTPUT:
            continue
        assert _outputs(command) == len([a for a in command.outputs if a.required])


def test_the_fit_returns_its_coefficients_rather_than_taking_the_process():
    # Without the output name this segfaulted inside BART's io bookkeeping,
    # on our call and on BART's own test case alike.
    times, echoes = _echoes()
    made = bt.mobafit(times, echoes, G=True, m=3)
    assert isinstance(made, torch.Tensor)
    # The R2S model fits three coefficients per voxel.  The data here is noise
    # and an exponential fitted to noise may well run away, so what is checked
    # is that an array came back at all -- which it did not before.
    assert 3 == made.reshape(-1, 8, 8).shape[0]


def test_the_fit_recovers_a_decay_it_was_given():
    n, echoes = 8, 7
    times = torch.tensor([1.6 * k for k in range(1, echoes + 1)], dtype=torch.complex64)
    r2s = 0.05
    signal = torch.stack([torch.exp(-r2s * t) for t in times]).to(torch.complex64)
    images = (signal.reshape(echoes, 1, 1, 1, 1, 1) * torch.ones(1, 1, 1, 1, n, n)).to(
        torch.complex64
    )
    made = bt.mobafit(times.reshape(echoes, 1, 1, 1, 1, 1), images, G=True, m=3)
    fitted = made.reshape(-1, n, n)
    torch.testing.assert_close(fitted[1].real, torch.full((n, n), r2s), rtol=1e-2, atol=1e-3)


def test_a_command_that_prints_its_answer_still_prints_it():
    # `estdelay` prints the delays when it is given no output name and writes
    # a quadratic fit when it is; passing one so that `mobafit` stops crashing
    # must not quietly turn the one into the other.
    traj = bt.traj(x=16, y=8, r=True)
    kspace = torch.randn(1, 8, 16, 1, dtype=torch.complex64)
    made = bt.estdelay(traj, kspace)
    assert isinstance(made, str)
    assert 3 == len(made.split(":"))


@pytest.mark.parametrize(
    "name", ["bench", "estdelay", "estshift", "ismrmrd", "measure", "roistat", "seq"]
)
def test_the_guarded_commands_are_left_alone(name):
    # Each of these writes through anon_cfl or behind an if, so BART is happy
    # with no name and answers with what it printed.  Asking for an output
    # would change what they return.
    assert name not in _WRITES_ITS_OPTIONAL_OUTPUT
    assert 0 == _outputs(COMMANDS[name])
