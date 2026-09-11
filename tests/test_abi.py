"""The seam between the header and ctypes.

``src/bartorch/_abi.py`` is generated from ``src/csrc/include/bartorch.h``.  A
signature written twice is a signature that drifts, and a wrong one is a
silently truncated pointer rather than an error, so what is checked here is
that the generated file is still what the header produces and that every entry
point the header exports has a signature at all.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build_tools"))

import gen_abi  # noqa: E402

# ctypes is the vocabulary of the ABI, and of resolving routines in libraries
# that are already in the process.  Nothing else in the package names it: above
# these modules the package works in shapes, integers and tensors, which is
# what would let the binding be something other than ctypes without a caller
# changing.
CTYPES_MODULES = {
    "_abi.py",  # the generated signatures
    "_lib.py",  # finding and loading libbartorch
    "_marshal.py",  # what the ABI's arguments look like
    "_backend.py",  # addresses of BLAS and LAPACK routines, in other libraries
    "_finufft.py",  # addresses of FINUFFT's entry points, in its own wheel
}


def test_the_generated_abi_is_what_the_header_produces():
    checked_in = gen_abi.OUTPUT.read_text()
    assert checked_in == gen_abi.generate(), (
        "src/bartorch/_abi.py is out of date with src/csrc/include/bartorch.h; "
        "run `python build_tools/gen_abi.py`"
    )


def test_every_entry_point_the_header_exports_has_a_signature():
    from bartorch import _abi
    from bartorch._lib import library

    exported = [name for name, _, _ in gen_abi.parse(gen_abi.HEADER.read_text())["functions"]]
    assert set(exported) == set(_abi.SYMBOLS)

    lib = library()
    for name in _abi.SYMBOLS:
        fn = getattr(lib, name, None)
        assert fn is not None, f"{name} is in the header but not in the library"
        # An unbound entry point returns a C int, which truncates a pointer.
        assert fn.argtypes is not None, f"{name} has no argtypes"
        assert hasattr(fn, "restype"), f"{name} has no restype"


def test_a_pointer_returning_entry_point_is_not_left_returning_an_int():
    """The failure this catches loses the top half of every address."""
    import ctypes

    from bartorch._lib import library

    lib = library()
    for name in ("bartorch_linop_fft", "bartorch_nlop_from_linop", "bartorch_host_alloc"):
        assert getattr(lib, name).restype is ctypes.c_void_p


@pytest.mark.parametrize(
    "path",
    sorted(p for p in (ROOT / "src" / "bartorch").rglob("*.py")),
    ids=lambda p: str(p.relative_to(ROOT / "src" / "bartorch")),
)
def test_ctypes_stays_in_the_modules_that_own_it(path):
    if path.name in CTYPES_MODULES:
        return
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(n == "ctypes" or n.startswith("ctypes.") for n in names), (
            f"{path.name} imports ctypes; reach the library through bartorch._lib "
            f"and bartorch._marshal instead, or add it to CTYPES_MODULES with a reason"
        )


def test_the_generator_refuses_a_type_it_has_not_been_taught():
    """It stops rather than guessing, which is what makes the output trustworthy."""
    header = """
    #define BARTORCH_DIMS 16
    #define BARTORCH_API
    BARTORCH_API int bartorch_thing(struct someone_elses_struct s);
    """
    with pytest.raises(gen_abi.Unknown):
        gen_abi.parse(header)


def test_bartorch_imports_without_naming_ctypes_at_the_top_level():
    """The package's own surface is shapes and tensors."""
    importlib.import_module("bartorch")
