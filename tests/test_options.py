"""What a BART option is called in Python, and what it reaches on the way out.

The flag is looked up in the catalogue rather than derived from the keyword,
because BART writes some long names with hyphens and others with underscores
and no rule over the keyword can be right for both.  Deriving it is what made
``pics(psf_export=...)`` send ``--psf-export``, which BART rejects.
"""

import pytest

from bartorch import _catalogue as catalogue
from bartorch import _options
from bartorch._dispatch import _flag_string, run_command


def test_every_option_of_every_command_reaches_its_own_flag():
    """All of them, because the ones that did not were not a special case."""
    missed = [
        (command.name, _options.python_name(option), option.flag)
        for command in catalogue.COMMANDS.values()
        for option in command.options
        if _flag_string(_options.python_name(option), command.name) != option.flag
    ]
    assert missed == []


@pytest.mark.parametrize(
    ("command", "keyword", "flag"),
    [
        # The long names BART spells with an underscore, which a keyword that
        # is turned back into a flag by rule cannot reach.
        ("pics", "psf_export", "--psf_export"),
        ("pics", "psf_import", "--psf_import"),
        ("pics", "ist_last", "--ist_last"),
        ("pics", "fista_pqr", "--fista_pqr"),
        ("moba", "scale_data", "--scale_data"),
        ("seq", "slice_thickness", "--slice_thickness"),
        # And the ones it does spell with a hyphen, which must still work.
        ("pics", "no_toeplitz", "--no-toeplitz"),
        ("pics", "gpu_gridding", "--gpu-gridding"),
        # A letter is a name too, and the word reaches the same option.
        ("pics", "U", "-U"),
        ("pics", "lowmem", "-U"),
    ],
)
def test_a_keyword_reaches_the_flag_bart_actually_has(command, keyword, flag):
    assert _flag_string(keyword, command) == flag


def test_a_digit_that_is_part_of_the_name_is_not_read_as_a_repetition():
    """``moba`` has ``--kfilter-1`` and ``--kfilter-2``; reading the digit as a
    repeat would send both to a ``--kfilter`` that does not exist."""
    assert _flag_string("kfilter_1", "moba") == "--kfilter-1"
    assert _flag_string("kfilter_2", "moba") == "--kfilter-2"


def test_a_digit_that_is_a_repetition_still_is_one():
    """``pics`` takes ``-R`` more than once, and there is no ``-R1``."""
    assert _flag_string("R_1", "pics") == "-R"
    assert _flag_string("R_2", "pics") == "-R"


def test_a_keyword_the_catalogue_does_not_know_still_gets_a_guess():
    """A flag passed through, or a command from a newer BART."""
    assert _flag_string("something_new", "pics") == "--something-new"
    assert _flag_string("x", "not_a_command") == "-x"


# --- asking for help --------------------------------------------------------


def test_asking_bart_for_help_is_refused_rather_than_obeyed():
    """BART answers by printing its usage and calling exit, and BART is in this
    process, so obeying ends the interpreter with no message."""
    with pytest.raises(ValueError, match="describe"):
        run_command(["pics", "-h"])
    with pytest.raises(ValueError, match="describe"):
        run_command(["pics", "--help"])


def test_the_help_is_answered_from_the_catalogue_instead():
    said = _options.describe("pics")
    assert said.startswith("pics -- Parallel-imaging")
    assert "kspace" in said and "sensitivities" in said
    assert "regularization parameter" in said


def test_the_description_shows_both_spellings_and_the_value_taken():
    said = _options.describe("pics")
    assert "lowmem  (-U)" in said
    assert "r=lambda" in said


def test_the_description_gathers_a_choice_into_one_line():
    said = _options.describe("pics")
    assert "one of: ist, fista, eulermaruyama, admm, pridu" in said


def test_describing_something_bart_does_not_have_says_so():
    with pytest.raises(KeyError, match="no command"):
        _options.describe("definitely_not_a_bart_command")


# --- the naming rules -------------------------------------------------------


def test_a_word_is_preferred_to_a_letter():
    lowmem = next(o for o in catalogue.COMMANDS["pics"].options if o.short == "U")
    assert _options.python_name(lowmem) == "lowmem"


def test_a_letter_is_kept_as_a_second_way_in():
    table = _options.options_by_name("pics")
    assert table["U"] is table["lowmem"]


def test_case_is_not_folded():
    """``pics`` has both ``-r`` and ``-R``, and they are different options."""
    table = _options.options_by_name("pics")
    assert table["r"].flag == "-r"
    assert table["R"].flag == "-R"
