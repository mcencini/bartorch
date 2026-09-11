"""That the library under test was built from the sources under test.

Nothing else here would notice.  The Python side is read from the checkout, so
an edit to it takes effect the moment a test imports it; the C side is a
shared object that was compiled at some point in the past, and a stale one
answers every call happily with the old behaviour.  What that looks like is a
test that passes for a fix that is not in the binary, or fails for a fix that
is -- which is worse than either.

Some of it is caught already.  ``_abi.py`` is generated from the header and
``tests/test_abi.py`` holds the two together, and every symbol the header
declares is looked up at load, so a header that grew an entry point the
library does not have fails at import.  What is left is everything that does
not change the set of symbols: an entry point whose arguments changed, and any
edit at all to a ``.c`` file.  For those there is nothing to compare but the
clock.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# The same definition `scripts/run_tests.sh` builds against, so that what it
# thinks is up to date and what this thinks is up to date cannot disagree.
import sources  # noqa: E402


def test_the_library_is_newer_than_the_sources_it_was_built_from():
    """Otherwise the suite is measuring a build that no longer exists.

    ``scripts/run_tests.sh`` builds before it runs, so the ordinary path never
    reaches this.  What reaches it is ``pytest tests/`` after a change to the C
    side, which is exactly when a green run means nothing.
    """
    from bartorch._lib import library_path

    source, changed = sources.newest()
    if source is None:
        pytest.skip("the C sources are not beside this checkout")

    library = library_path()
    built = library.stat().st_mtime
    assert built >= changed, (
        f"{library} was built before {source.relative_to(ROOT)} was last changed "
        f"({built:.0f} < {changed:.0f}), so these tests are running against a stale "
        "library; rebuild with `./scripts/run_tests.sh` or `cmake --build <dir>`"
    )
