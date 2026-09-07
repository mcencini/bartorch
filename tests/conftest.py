"""State the library carries between tests.

The FINUFFT substitution is a process-wide setting that installs itself the
first time anything needs it, so a test that turns it off, loosens its
tolerance or lets BART's gridder answer would otherwise hand that on to
whatever runs next.  This puts it back.
"""

import pytest

from bartorch import _finufft


@pytest.fixture(autouse=True)
def _finufft_defaults():
    yield
    if _finufft.available():
        try:
            _finufft.use_in_tools(True)
        except (ImportError, RuntimeError):
            pass
