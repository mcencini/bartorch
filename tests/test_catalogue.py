"""The catalogue is what BART's sources say, and stays that way.

``src/bartorch/_catalogue.py`` is generated from the BART submodule.  It
is a description of BART rather than a Python API on purpose: the wrappers
people call are written by hand against it, so a name in Python is chosen
rather than transliterated, and a command nobody has wrapped is still
countable.

What is checked here is that it is current, that it covers every tool BART
builds, and that it keeps the things a straight transliteration of the command
line throws away -- which is most of why it exists.
"""

import re
import sys
from pathlib import Path

import pytest

from bartorch import _catalogue as catalogue

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gen_catalogue  # noqa: E402

BART_SRC = ROOT / "external" / "bart" / "src"
needs_bart = pytest.mark.skipif(
    not (BART_SRC / "bart.c").exists(),
    reason="the BART submodule is not checked out",
)


@needs_bart
def test_the_catalogue_is_what_barts_sources_say():
    assert gen_catalogue.OUTPUT.read_text() == gen_catalogue.generate(), (
        "src/bartorch/_catalogue.py is out of date with the BART submodule; "
        "run `python scripts/gen_catalogue.py`"
    )


@needs_bart
def test_every_tool_bart_builds_is_in_it():
    built = set()
    for path in BART_SRC.glob("*.c"):
        match = re.search(r"^int main_(\w+)", path.read_text(errors="replace"), re.M)
        if match:
            built.add(match.group(1))
    assert built == set(catalogue.NAMES)


def test_it_is_not_empty_and_every_command_has_a_description():
    assert len(catalogue.COMMANDS) > 100
    for command in catalogue.COMMANDS.values():
        assert command.name
        assert command.help, f"{command.name} has no description"


def test_every_option_can_be_spelled():
    for command in catalogue.COMMANDS.values():
        for option in command.options:
            assert option.short or option.long, f"{command.name} has an unnameable option"
            assert option.flag.startswith("-")


# --- what a transliteration of the command line loses -----------------------


def test_a_long_name_survives_beside_the_short_one():
    """Taking the letter and dropping the word is how `pics` ends up with an
    argument called ``U`` instead of ``lowmem``."""
    both = {
        (o.short, o.long) for o in catalogue.COMMANDS["pics"].options if o.short and o.long
    }
    assert ("U", "lowmem") in both
    assert ("m", "admm") in both


def test_an_option_with_no_letter_at_all_is_kept():
    long_only = [o for o in catalogue.COMMANDS["pics"].options if not o.short]
    assert len(long_only) > 10
    assert "no-toeplitz" in {o.long for o in long_only}


def test_a_run_of_select_options_is_one_choice():
    """BART writes a choice as several options into one variable.  Read one at a
    time they are unrelated flags; grouped, they are the solver."""
    choices = catalogue.COMMANDS["pics"].choices()
    assert len(choices) == 1
    (arms,) = choices.values()
    spellings = {o.long or o.short for o in arms}
    assert {"ist", "fista", "admm", "pridu"} <= spellings


def test_select_and_subopt_are_read_at_all():
    """Neither had ever been read, so 132 options and 52 sub-tables across BART
    were invisible from Python."""
    kinds = [o.kind for c in catalogue.COMMANDS.values() for o in c.options]
    assert kinds.count("SELECT") > 100
    assert kinds.count("SUBOPT") > 40


def test_an_optional_output_is_an_output_and_not_a_path():
    """``pics --psf_export`` writes an array.  A wrapper that takes it as a
    string takes a filename, which is the one thing this package does not have."""
    exports = [o for o in catalogue.COMMANDS["pics"].options if o.long == "psf_export"]
    assert exports and exports[0].kind == "OUTFILE"


def test_a_metavar_is_not_a_description():
    """``OPT_FLVECN`` takes no metavar, one argument fewer than the options
    beside it, and reading it like them puts the description where the metavar
    goes -- which is silent, and shows up only in a docstring nobody reads.
    A metavar is a word or two; a description is a sentence."""
    for command in catalogue.COMMANDS.values():
        for option in command.options:
            assert len(option.arg) < 32, f"{command.name} {option.flag}: {option.arg!r}"


def test_an_option_bart_writes_out_by_hand_is_read_too():
    """``pics -R`` is a brace initialiser rather than a macro, and it is the
    regularization option people reach for first."""
    by_flag = {o.flag: o for o in catalogue.COMMANDS["pics"].options}
    assert "-R" in by_flag
    assert by_flag["-R"].arg == "<T>:A:B:C"
    assert "regularization" in by_flag["-R"].help


def test_nothing_is_dropped_without_saying_so():
    """Whatever the reader cannot make sense of is named, so that a construct
    BART starts using fails here instead of leaving an option missing."""
    assert set(catalogue.UNREAD) == {"mobafit", "nlinv"}
    # Two of mobafit's are inside `#if 0`, so BART does not compile them
    # either; nlinv's picks its letter from the compatibility version.
    assert len(catalogue.UNREAD["mobafit"]) == 2
    assert len(catalogue.UNREAD["nlinv"]) == 1


# --- the shape a wrapper reads off it ---------------------------------------


def test_a_command_says_what_it_reads_and_writes():
    pics = catalogue.COMMANDS["pics"]
    assert [a.name for a in pics.inputs] == ["kspace", "sensitivities"]
    assert [a.name for a in pics.outputs] == ["output"]


def test_a_positional_value_is_told_apart_from_an_array():
    """``bart phantom`` has no input array; ``bart resize`` takes numbers among
    its positional arguments, and those are not tensors."""
    phantom = catalogue.COMMANDS["phantom"]
    assert phantom.inputs == ()
    assert len(phantom.outputs) == 1
    for command in catalogue.COMMANDS.values():
        for argument in command.arguments:
            assert argument.is_array == (argument.kind in ("INFILE", "OUTFILE", "INOUTFILE"))


def test_a_complex_scalar_on_the_command_line_is_not_an_array():
    """``bart scale <factor> <input> <output>`` reads the factor from argv.
    Taking it for an array would hand BART a tensor where it wants a number."""
    for name in ("scale", "saxpy", "spow"):
        scalar = catalogue.COMMANDS[name].arguments[0]
        assert scalar.kind == "CFL"
        assert not scalar.is_array
        assert scalar not in catalogue.COMMANDS[name].inputs
        assert scalar in catalogue.COMMANDS[name].values


def test_no_command_takes_a_value_after_an_array():
    """The argument vector is built as flags, then values, then arrays, so a
    command that wanted one in between could not be called at all."""
    for command in catalogue.COMMANDS.values():
        seen_array = False
        for argument in command.arguments:
            if argument.kind == "OUTFILE":
                continue
            if argument.is_array:
                seen_array = True
            else:
                assert not seen_array, f"{command.name} takes {argument.name} after an array"


def test_a_flag_that_takes_no_value_says_so():
    pics = catalogue.COMMANDS["pics"]
    by_flag = {o.flag: o for o in pics.options}
    assert not by_flag["-g"].takes_value  # OPT_SET
    assert by_flag["-r"].takes_value  # OPT_FLOAT
    assert not by_flag["--fista"].takes_value  # OPTL_SELECT


def test_the_version_it_was_read_from_is_recorded():
    """A submodule bump that is not regenerated should be visible."""
    assert catalogue.BART_VERSION
    stated = ROOT / "external" / "bart" / "version.txt"
    if stated.exists():
        assert catalogue.BART_VERSION == stated.read_text().strip()
