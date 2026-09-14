"""Reading a Mach-O's load commands, and deciding whether the two OpenMP runtimes are one.

``scripts/macos_openmp.py`` decides whether FINUFFT's library can be pointed
at the OpenMP runtime torch carries.  What it decides from is the load
commands, so the reader is held against Mach-O files built here rather than
against a checked-in binary, and the decision is held against every answer it
can give.  The patch itself needs macOS and a toolchain, and is what the
macOS CI job exercises.
"""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "macos_openmp.py"


def _module():
    spec = importlib.util.spec_from_file_location("macos_openmp", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("macos_openmp", module)
    spec.loader.exec_module(module)
    return module


mo = _module()


# --- Mach-O files to read -----------------------------------------------------


def _version(text: str) -> int:
    major, minor, patch = (int(p) for p in text.split("."))
    return (major << 16) | (minor << 8) | patch


def _dylib_command(cmd: int, name: str, compat: str, current: str) -> bytes:
    """One LC_ID_DYLIB or LC_LOAD_DYLIB, with its name past the fixed fields."""
    payload = name.encode() + b"\x00"
    payload += b"\x00" * (-len(payload) % 8)
    size = 24 + len(payload)
    return struct.pack("<IIIIII", cmd, size, 24, 0, _version(current), _version(compat)) + payload


def _rpath_command(path: str) -> bytes:
    payload = path.encode() + b"\x00"
    payload += b"\x00" * (-len(payload) % 8)
    return struct.pack("<III", mo.LC_RPATH, 12 + len(payload), 12) + payload


def macho(tmp_path: Path, name: str, *commands: bytes, fat: bool = False) -> Path:
    """A 64-bit Mach-O carrying exactly ``commands``, thin or universal."""
    body = b"".join(commands)
    header = struct.pack(
        "<IiiIIIII", mo.MH_MAGIC_64, 0x0100000C, 0, 0x6, len(commands), len(body), 0, 0
    )
    image = header + body

    path = tmp_path / name
    if not fat:
        path.write_bytes(image)
        return path

    # One slice, at an offset the fat header points to.
    offset = 4096
    fat_header = struct.pack(">II", mo.FAT_MAGIC, 1)
    fat_header += struct.pack(">iiIII", 0x0100000C, 0, offset, len(image), 12)
    path.write_bytes(fat_header.ljust(offset, b"\x00") + image)
    return path


def test_the_reader_is_otool_without_the_toolchain(tmp_path):
    path = macho(
        tmp_path,
        "libfinufft.dylib",
        _dylib_command(mo.LC_ID_DYLIB, "@rpath/libfinufft.dylib", "0.0.0", "0.0.0"),
        _dylib_command(mo.LC_LOAD_DYLIB, "/usr/lib/libSystem.B.dylib", "1.0.0", "1345.120.2"),
        _dylib_command(mo.LC_LOAD_DYLIB, "@loader_path/.dylibs/libomp.dylib", "5.0.0", "5.0.0"),
        _rpath_command("@loader_path/"),
    )
    read = mo.read_macho(path)
    assert read.install_name == "@rpath/libfinufft.dylib"
    assert read.rpaths == ["@loader_path/"]
    assert read.loads == [
        ("/usr/lib/libSystem.B.dylib", "1.0.0", "1345.120.2"),
        ("@loader_path/.dylibs/libomp.dylib", "5.0.0", "5.0.0"),
    ]
    assert read.openmp_load() == ("@loader_path/.dylibs/libomp.dylib", "5.0.0", "5.0.0")


def test_a_compatibility_version_is_not_the_current_one(tmp_path):
    """The two sit next to each other in the load command and mean opposite things."""
    path = macho(
        tmp_path,
        "libx.dylib",
        _dylib_command(mo.LC_LOAD_DYLIB, "/usr/lib/libc++.1.dylib", "1.0.0", "1700.255.5"),
    )
    assert mo.read_macho(path).loads == [("/usr/lib/libc++.1.dylib", "1.0.0", "1700.255.5")]


def test_a_universal_binary_is_read_through_its_first_slice(tmp_path):
    path = macho(
        tmp_path,
        "libfat.dylib",
        _dylib_command(mo.LC_ID_DYLIB, "/opt/llvm-openmp/lib/libomp.dylib", "5.0.0", "5.0.0"),
        fat=True,
    )
    read = mo.read_macho(path)
    assert read.install_name == "/opt/llvm-openmp/lib/libomp.dylib"
    assert read.offers == "5.0.0"


def test_something_that_is_not_a_macho_is_refused(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"this is not a Mach-O, it is a sentence about one")
    with pytest.raises(ValueError, match="Mach-O"):
        mo.read_macho(path)


# --- what the runtimes are ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("@loader_path/.dylibs/libomp.dylib", "libomp"),
        ("/opt/intel/lib/libiomp5.dylib", "libiomp5"),
        ("libgomp.1.dylib", "libgomp"),
        ("/usr/lib/libSystem.B.dylib", None),
        # A name that merely starts with one, which is a different library.
        ("libompanion.dylib", None),
    ],
)
def test_an_openmp_runtime_is_told_from_its_name(name, family):
    assert mo._openmp_family(name) == family


# --- the decision -------------------------------------------------------------


def _omp(install_name="/opt/llvm-openmp/lib/libomp.dylib", offers="5.0.0"):
    return mo.Dylib(path=Path("libomp.dylib"), install_name=install_name, offers=offers)


def _finufft(loads=None, rpaths=()):
    return mo.Dylib(
        path=Path("libfinufft.dylib"),
        loads=list(
            loads
            if loads is not None
            else [("@loader_path/.dylibs/libomp.dylib", "5.0.0", "5.0.0")]
        ),
        rpaths=list(rpaths),
    )


RPATH = "@loader_path/../torch/lib"


def test_one_runtime_at_one_version_is_patched():
    decision = mo.decide(_omp(), _finufft(), _omp(), RPATH)
    assert decision.action == "patch"
    assert decision.change_from == "@loader_path/.dylibs/libomp.dylib"
    assert decision.add_rpath == RPATH


def test_a_library_already_pointed_at_torchs_copy_is_left_alone():
    """What makes rerunning it safe, and what the CI step relies on."""
    patched = _finufft(loads=[("@rpath/libomp.dylib", "5.0.0", "5.0.0")], rpaths=[RPATH])
    assert mo.decide(_omp(), patched, _omp(), RPATH).action == "already"


def test_the_same_name_without_the_rpath_is_not_already_done():
    """An @rpath entry that nothing resolves would be a broken library, not a patched one."""
    half = _finufft(loads=[("@rpath/libomp.dylib", "5.0.0", "5.0.0")], rpaths=[])
    assert mo.decide(_omp(), half, _omp(), RPATH).action != "already"


def test_two_different_runtimes_are_refused_rather_than_patched():
    """The case the request named: torch on Intel's runtime and FINUFFT on LLVM's."""
    intel = _omp(install_name="/opt/intel/lib/libiomp5.dylib")
    decision = mo.decide(intel, _finufft(), _omp(), RPATH)
    assert decision.action == "refuse"
    assert "different runtimes" in decision.reason


def test_the_same_runtime_at_a_version_the_loader_would_reject_is_refused():
    older = _omp(offers="4.0.0")
    decision = mo.decide(older, _finufft(), _omp(), RPATH)
    assert decision.action == "refuse"
    assert "compatibility version" in decision.reason


def test_a_library_that_loads_no_openmp_runtime_is_refused():
    plain = _finufft(loads=[("/usr/lib/libSystem.B.dylib", "1.0.0", "1.0.0")])
    decision = mo.decide(_omp(), plain, None, RPATH)
    assert decision.action == "refuse"
    assert "nothing to point" in decision.reason


# --- where the runtime is reached from ----------------------------------------


def test_the_rpath_is_relative_so_the_environment_can_move():
    """An absolute path into site-packages would not survive a rename."""
    site = Path("/opt/env/lib/python3.12/site-packages")
    assert (
        mo.rpath_for(site / "finufft" / "libfinufft.dylib", site / "torch" / "lib" / "libomp.dylib")
        == RPATH
    )


def test_packages_that_are_not_siblings_still_get_a_path_that_reaches():
    site = Path("/opt/env/site-packages")
    elsewhere = Path("/opt/other/torch/lib/libomp.dylib")
    rpath = mo.rpath_for(site / "finufft" / "libfinufft.dylib", elsewhere)
    assert rpath.startswith("@loader_path/")
    assert rpath.endswith("/torch/lib")


# --- the layout on disk -------------------------------------------------------


def test_the_layout_is_found_from_an_unpacked_pair(tmp_path):
    (tmp_path / "torch" / "lib").mkdir(parents=True)
    (tmp_path / "finufft" / ".dylibs").mkdir(parents=True)
    for path in (
        tmp_path / "torch" / "lib" / "libomp.dylib",
        tmp_path / "finufft" / "libfinufft.dylib",
        tmp_path / "finufft" / ".dylibs" / "libomp.dylib",
    ):
        path.write_bytes(b"")

    layout = mo.find_layout(tmp_path)
    assert layout.torch_omp.name == "libomp.dylib"
    assert layout.finufft_lib.name == "libfinufft.dylib"
    assert layout.finufft_omp is not None


def test_a_finufft_without_its_library_says_so(tmp_path):
    (tmp_path / "torch" / "lib").mkdir(parents=True)
    (tmp_path / "torch" / "lib" / "libomp.dylib").write_bytes(b"")
    (tmp_path / "finufft").mkdir()
    with pytest.raises(SystemExit, match="libfinufft"):
        mo.find_layout(tmp_path)
