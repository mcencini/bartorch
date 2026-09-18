"""``bartorch.cli`` against the command line it serves.

A script that calls ``bart`` has to run against ``bartorch`` unchanged, so the
question a test here answers is whether the two answer the same bits: the app
route against the command route, over the same argv and the same files.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import bartorch.tools as bt
from bartorch._dispatch import run_command
from bartorch.cli import _argv, main, route
from bartorch.io import readcfl, writecfl

SIZE, COILS, ACCEL = 24, 4, 2


@pytest.fixture
def _dataset(tmp_path, monkeypatch):
    """An under-sampled k-space and its sensitivities, as CFL pairs."""
    monkeypatch.chdir(tmp_path)
    kspace = bt.phantom(SIZE, coils=COILS, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    mask = torch.zeros(SIZE, dtype=torch.complex64)
    mask[::ACCEL] = 1
    mask[SIZE // 2 - 2 : SIZE // 2 + 2] = 1
    writecfl("ksp", (kspace * mask.reshape(SIZE, 1)).numpy().T)
    writecfl("maps", maps.numpy().T)
    return tmp_path


#: Command lines a script would really contain, each run both ways.
_LINES = [
    ["pics", "-l1", "-r", "0.01", "-i", "20"],
    ["pics", "-R", "W:7:0:0.01", "-i", "20"],
    ["pics", "-R", "T:6:0:0.005", "-i", "20"],
    ["pics", "-r", "0.1", "-i", "20"],
    ["pics", "-l2", "-r", "0.05", "-i", "20"],
    ["pics", "-R", "W:3:0:0.01", "-i", "20"],
    ["pocsense", "-i", "10"],
    ["pocsense", "-i", "10", "-r", "0.01", "-l", "1"],
]


@pytest.mark.parametrize("line", _LINES, ids=[" ".join(line) for line in _LINES])
def test_the_app_route_answers_what_the_command_answers(line, _dataset):
    where, _ = route(line[0], [*line[1:], "ksp", "maps", "x"])
    assert where == "app", f"{' '.join(line)} did not reach the app"

    assert main([*line, "ksp", "maps", "out_app"]) == 0
    code, _, failure = run_command([*line, "ksp", "maps", "out_cmd"])
    assert code == 0, failure

    assert np.array_equal(readcfl("out_app"), readcfl("out_cmd"))


def test_an_argument_the_reader_cannot_express_goes_to_the_command(_dataset):
    """``pics -c`` is the real-value constraint, which the app does not take;
    the command does, so it runs and the answer is still BART's."""
    where, _ = route("pics", ["-c", "-i", "10", "ksp", "maps", "x"])
    assert where == "command"

    assert main(["pics", "-c", "-i", "10", "ksp", "maps", "out"]) == 0
    assert readcfl("out").shape == (SIZE, SIZE)


def test_a_command_with_no_app_runs_as_itself(_dataset):
    assert route("fft", ["-i", "6", "ksp", "img"]) == ("command", None)
    assert main(["fft", "-i", "6", "ksp", "img"]) == 0
    assert readcfl("img").shape == (SIZE, SIZE, 1, COILS)


def test_an_input_that_is_not_there_is_named_before_bart_is_asked(_dataset, capsys):
    """BART would report it, and would be the one to ask -- except that a
    command which fails while loading its arguments leaves the library unable
    to serve the next call in the same process."""
    assert main(["pics", "nosuchfile", "maps", "out"]) == 1
    assert "no such input: nosuchfile" in capsys.readouterr().err

    assert main(["pics", "-t", "nosuchtraj", "ksp", "maps", "out"]) == 1
    assert "no such input: nosuchtraj" in capsys.readouterr().err

    # And the library still answers, which is the whole point.
    assert main(["fft", "-i", "6", "ksp", "img"]) == 0


def test_an_unknown_command_is_refused(capsys):
    assert main(["nosuchcommand"]) == 1
    assert "no command called" in capsys.readouterr().err


def test_help_comes_from_the_catalogue(capsys):
    """BART answers its own help by calling ``exit``, which in this process
    would end the interpreter."""
    assert main(["pocsense", "--help"]) == 0
    printed = capsys.readouterr().out
    assert "Perform POCSENSE reconstruction" in printed
    assert "max. number of iterations" in printed


def test_the_listing_names_every_command(capsys):
    from bartorch._catalogue import COMMANDS

    assert main([]) == 0
    printed = capsys.readouterr().out
    for name in ("pics", "ecalib", "nufft", "traj"):
        assert f"    {name}" in printed
    assert "bart " not in printed.split("Commands")[0].replace("`bart <command>`", "")
    assert len(COMMANDS) > 50


# --- the reader ------------------------------------------------------------


def test_short_flags_are_read_the_way_getopt_reads_them():
    """A value stuck to its letter, and value-less letters written together."""
    options, files = _argv.parse("pics", ["-i30", "-eS", "-r", "0.01", "k", "m", "o"])
    assert options == {"i": [30], "e": [True], "S": [True], "r": [0.01]}
    assert files == ["k", "m", "o"]


def test_long_flags_take_their_value_either_way():
    joined, _ = _argv.parse("pics", ["--wavelet=haar", "k", "m", "o"])
    apart, _ = _argv.parse("pics", ["--wavelet", "haar", "k", "m", "o"])
    assert joined == apart == {"wavelet": ["haar"]}


def test_a_selector_says_which_arm_was_given():
    options, _ = _argv.parse("pics", ["--fista", "k", "m", "o"])
    assert options == {"fista": ["--fista"]}
    letter, _ = _argv.parse("pics", ["-I", "k", "m", "o"])
    assert letter == {"ist": ["-I"]}


def test_an_option_the_command_does_not_have_is_unsupported():
    with pytest.raises(_argv.Unsupported, match="not an option"):
        _argv.parse("pics", ["--nosuchoption", "k", "m", "o"])


def test_a_regularizer_string_is_the_term_it_names():
    from bartorch import priors

    term = _argv.regularizer("W:7:0:0.01", ndim=2)
    assert isinstance(term, priors.Wavelet)
    assert term.weight == 0.01

    assert isinstance(_argv.regularizer("Q:0.1", ndim=2), priors.L2)
    assert isinstance(_argv.regularizer("S", ndim=2), priors.NonNegative)
    assert isinstance(_argv.regularizer("N:3:0:12", ndim=2), priors.ImageNIHT)


def test_a_bitmask_names_the_axes_the_image_has():
    """BART carries a two-dimensional image on three axes and filters the third
    out with ``md_nontriv_dims``; the image here does not have it at all, so
    ``-R W:7`` and ``-R W:3`` are the same term on a slice."""
    assert _argv.axes(7, 3) == (-1, -2, -3)
    assert _argv.axes(7, 2) == (-1, -2)
    assert _argv.axes(3, 2) == (-1, -2)
    assert _argv.axes(6, 3) == (-2, -3)


def test_a_term_this_package_does_not_offer_is_unsupported():
    with pytest.raises(_argv.Unsupported, match="not a term"):
        _argv.regularizer("TF:{graph}:0.1", ndim=2)
