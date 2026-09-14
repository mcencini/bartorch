#!/usr/bin/env python3
"""Make torch's and FINUFFT's OpenMP runtimes one runtime, on macOS.

    python scripts/macos_openmp.py diagnose    # read only; says what it would do
    python scripts/macos_openmp.py patch       # repoint FINUFFT, then re-sign
    python scripts/macos_openmp.py verify      # the patch held, and both still run

Both wheels carry their own copy of LLVM's OpenMP runtime -- torch at
``torch/lib/libomp.dylib`` and FINUFFT at ``finufft/.dylibs/libomp.dylib`` --
and LLVM's runtime ends the process rather than run beside a second copy of
itself (``OMP: Error #15``).  So ``bartorch._finufft`` declines the
substitution when it sees two of them loaded, and BART's own gridder serves
the transforms instead.

This points FINUFFT's library at the copy torch carries, so that one runtime
is loaded and the substitution can be taken up with OpenMP left on.  That is
not what ``KMP_DUPLICATE_LIB_OK=TRUE`` does: the flag tells one runtime to
tolerate a second live copy, with two thread pools and two sets of
thread-local state behind it, which the runtime's own authors document as
unsafe.  Here there is one copy, and torch and FINUFFT share its pool.

It rewrites a file inside another package, so ``pip install -U finufft``
undoes it and it has to be run again.  It is written to be safe to rerun: a
library already pointing at torch's copy is left alone.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path

#: Mach-O load commands this reads; the rest are skipped by their length.
LC_ID_DYLIB = 0x0D
LC_LOAD_DYLIB = 0x0C
LC_RPATH = 0x8000001C

MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
FAT_MAGIC = 0xCAFEBABE
FAT_CIGAM = 0xBEBAFECA

#: What an OpenMP runtime's library is called, whoever built it.  LLVM's and
#: Intel's are ABI-compatible in principle and are still not mixed here: the
#: point is one runtime, and telling which of two a symbol came from is not
#: something a patch can promise.
OPENMP_NAMES = ("libomp", "libiomp5", "libgomp")


@dataclass
class Dylib:
    """One Mach-O image's load commands, as ``otool -L`` and ``-l`` show them."""

    path: Path
    install_name: str = ""
    #: The compatibility version this image itself offers, from LC_ID_DYLIB.
    offers: str = ""
    #: ``(name, compatibility version, current version)`` per LC_LOAD_DYLIB.
    loads: list[tuple[str, str, str]] = field(default_factory=list)
    rpaths: list[str] = field(default_factory=list)

    def openmp_load(self) -> tuple[str, str, str] | None:
        """The entry this image loads an OpenMP runtime through, if it has one."""
        for load in self.loads:
            if _openmp_family(load[0]) is not None:
                return load
        return None


def _version(value: int) -> str:
    """A Mach-O dylib version, which is packed as ``xxxx.yy.zz``."""
    return f"{(value >> 16) & 0xFFFF}.{(value >> 8) & 0xFF}.{value & 0xFF}"


def _openmp_family(name: str) -> str | None:
    """Which OpenMP runtime a dylib path names, or ``None``."""
    base = name.rsplit("/", 1)[-1]
    for family in OPENMP_NAMES:
        if base.startswith(family + "."):
            return family
    return None


def read_macho(path: Path) -> Dylib:
    """The load commands of a Mach-O file, without a toolchain.

    Reads the first 64-bit slice of a universal binary, which is enough:
    the wheels' slices carry the same dependencies, and what this decides is
    which library is loaded rather than for which architecture.  Raises
    ``ValueError`` on anything that is not a Mach-O.
    """
    raw = path.read_bytes()
    offset, endian = _first_slice(raw, path)

    magic, _cputype, _cpusub, _filetype, ncmds = struct.unpack_from(endian + "IiiII", raw, offset)
    if magic not in (MH_MAGIC_64, MH_CIGAM_64):
        raise ValueError(f"{path} is not a 64-bit Mach-O")

    out = Dylib(path=path)
    at = offset + 32  # the 64-bit header, including its reserved word
    for _ in range(ncmds):
        cmd, size = struct.unpack_from(endian + "II", raw, at)
        if cmd in (LC_ID_DYLIB, LC_LOAD_DYLIB):
            name_off, _ts, current, compat = struct.unpack_from(endian + "IIII", raw, at + 8)
            name = _string_at(raw, at + name_off, size - name_off)
            if cmd == LC_ID_DYLIB:
                out.install_name = name
                out.offers = _version(compat)
            else:
                out.loads.append((name, _version(compat), _version(current)))
        elif cmd == LC_RPATH:
            name_off = struct.unpack_from(endian + "I", raw, at + 8)[0]
            out.rpaths.append(_string_at(raw, at + name_off, size - name_off))
        at += size
    return out


def _first_slice(raw: bytes, path: Path) -> tuple[int, str]:
    """``(offset, struct endianness)`` of the slice to read."""
    if len(raw) < 8:
        raise ValueError(f"{path} is too short to be a Mach-O")
    magic = struct.unpack_from(">I", raw, 0)[0]
    if magic in (FAT_MAGIC, FAT_CIGAM):
        end = ">" if magic == FAT_MAGIC else "<"
        count = struct.unpack_from(end + "I", raw, 4)[0]
        if count < 1:
            raise ValueError(f"{path} is a universal binary with no slices")
        # fat_arch: cputype, cpusubtype, offset, size, align
        slice_off = struct.unpack_from(end + "iiIII", raw, 8)[2]
        return slice_off, _endian_at(raw, slice_off, path)
    return 0, _endian_at(raw, 0, path)


def _endian_at(raw: bytes, offset: int, path: Path) -> str:
    magic = struct.unpack_from("<I", raw, offset)[0]
    if magic == MH_MAGIC_64:
        return "<"
    if magic == MH_CIGAM_64:
        return ">"
    raise ValueError(f"{path} is not a 64-bit Mach-O")


def _string_at(raw: bytes, start: int, length: int) -> str:
    return raw[start : start + length].split(b"\x00", 1)[0].decode(errors="replace")


# --- what is installed --------------------------------------------------------


@dataclass
class Layout:
    """Where the two packages and their OpenMP runtimes are."""

    torch_omp: Path
    finufft_lib: Path
    finufft_omp: Path | None


def find_layout(site: Path | None = None) -> Layout:
    """The files this works on, found through the installed packages themselves.

    ``site`` overrides where to look, which is what lets this be exercised
    against an unpacked wheel rather than an installed one.
    """
    if site is not None:
        torch_dir, finufft_dir = site / "torch", site / "finufft"
    else:
        torch_dir, finufft_dir = _package_dir("torch"), _package_dir("finufft")

    torch_omp = torch_dir / "lib" / "libomp.dylib"
    if not torch_omp.exists():
        found = sorted(torch_dir.glob("lib/lib*omp*.dylib"))
        if not found:
            raise SystemExit(f"torch carries no OpenMP runtime under {torch_dir / 'lib'}")
        torch_omp = found[0]

    finufft_lib = finufft_dir / "libfinufft.dylib"
    if not finufft_lib.exists():
        raise SystemExit(f"no libfinufft.dylib under {finufft_dir}")

    bundled = sorted(finufft_dir.glob(".dylibs/lib*omp*.dylib"))
    return Layout(torch_omp, finufft_lib, bundled[0] if bundled else None)


def _package_dir(name: str) -> Path:
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"{name} is not installed in this interpreter")
    return Path(next(iter(spec.submodule_search_locations)))


# --- the decision -------------------------------------------------------------


@dataclass
class Decision:
    """What to do about the pair, and why.

    ``action`` is ``"patch"``, ``"already"`` where FINUFFT resolves to torch's
    runtime as it stands, or ``"refuse"`` where the two cannot be made one.
    """

    action: str
    reason: str
    change_from: str = ""
    add_rpath: str = ""


def decide(torch_omp: Dylib, finufft: Dylib, bundled: Dylib | None, rpath: str) -> Decision:
    """Whether FINUFFT can be pointed at torch's OpenMP runtime.

    Refuses rather than patches wherever the two runtimes are not the same
    one -- a different implementation, or the same one at an incompatible
    version -- because a patch cannot make those interchangeable and a
    half-compatible runtime fails at a call rather than at load.
    """
    load = finufft.openmp_load()
    if load is None:
        return Decision("refuse", "libfinufft.dylib loads no OpenMP runtime; nothing to point")

    name, compat, _current = load
    if name.startswith("@rpath/") and rpath in finufft.rpaths:
        return Decision("already", f"libfinufft.dylib already resolves {name} through {rpath}")

    theirs = _openmp_family(torch_omp.install_name or torch_omp.path.name)
    ours = _openmp_family(name)
    if theirs != ours:
        return Decision(
            "refuse",
            f"torch carries {theirs} and FINUFFT wants {ours}; these are different runtimes, "
            "and pointing one at the other is not a change this can make safely",
        )

    if bundled is None:
        return Decision("refuse", f"FINUFFT loads {name} but carries no runtime to compare")

    if torch_omp.offers != compat:
        return Decision(
            "refuse",
            f"FINUFFT was built against {ours} compatibility version {compat} and torch carries "
            f"{torch_omp.offers}; the dynamic loader would reject the pair",
        )

    return Decision("patch", f"both carry {ours} {compat}", change_from=name, add_rpath=rpath)


def rpath_for(finufft_lib: Path, torch_omp: Path) -> str:
    """The ``LC_RPATH`` that reaches torch's runtime from FINUFFT's library.

    Relative to ``@loader_path`` so that the environment can be moved or
    renamed, which an absolute path into site-packages would not survive.
    """
    relative = os.path.relpath(torch_omp.parent, finufft_lib.parent)
    return f"@loader_path/{relative}"


# --- doing it -----------------------------------------------------------------


def apply(layout: Layout, decision: Decision, *, dry_run: bool = False) -> None:
    """Repoint FINUFFT's library and re-sign it.

    Re-signing is not optional on Apple Silicon: changing a load command
    invalidates the signature, and whether ``install_name_tool`` restores an
    ad-hoc one depends on the toolchain, so it is done here and checked.
    """
    if not dry_run:
        for tool in ("install_name_tool", "codesign"):
            if shutil.which(tool) is None:
                raise SystemExit(f"{tool} is not on PATH; install the Xcode command line tools")

    library = str(layout.finufft_lib)
    steps = [
        ["install_name_tool", "-change", decision.change_from, "@rpath/libomp.dylib", library],
    ]
    # -add_rpath fails on a duplicate, so it is asked for only where the entry
    # is not already there: what makes a second run a no-op rather than an error.
    if decision.add_rpath not in read_macho(layout.finufft_lib).rpaths:
        steps.append(["install_name_tool", "-add_rpath", decision.add_rpath, library])
    steps += [
        ["codesign", "--force", "--sign", "-", library],
        ["codesign", "--verify", library],
    ]

    for step in steps:
        print("  " + " ".join(step))
        if not dry_run:
            subprocess.run(step, check=True)


# --- what the patch is for ----------------------------------------------------

#: Counts the OpenMP runtimes the process has loaded, whichever order the two
#: packages were imported in.  One is the whole point; two is what ends the
#: process, and zero would mean neither library was reached at all.
LOADED = """
import ctypes as c, sys
{imports}
dyld = c.CDLL(None)
dyld._dyld_image_count.restype = c.c_uint32
dyld._dyld_get_image_name.restype = c.c_char_p
dyld._dyld_get_image_name.argtypes = [c.c_uint32]
names = [dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count())]
omp = [n.decode() for n in names if n and n.rsplit(b"/", 1)[-1].startswith((b"libomp.", b"libiomp5."))]
print(len(omp))
for path in omp:
    print(" ", path)
"""

#: Both runtimes have to be started, not merely loaded: an image can sit in
#: the process without its runtime having initialized, and it is the second
#: initialization that aborts.
WORKLOADS = """
import numpy as np, torch
torch.set_num_threads(4)
a = torch.randn(512, 512)
print("torch", float((a @ a).sum()) == float((a @ a).sum()))
import finufft
x = np.random.rand(20000) * 2 - 1
y = np.random.rand(20000) * 2 - 1
c = (np.random.rand(20000) + 1j * np.random.rand(20000)).astype(np.complex128)
f = finufft.nufft2d1(x, y, c, (64, 64), nthreads=4)
print("finufft", f.shape == (64, 64) and np.isfinite(f).all())
"""


def _probe(name: str, source: str) -> bool:
    print(f"\n{name}")
    done = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    for line in (done.stdout + done.stderr).splitlines():
        print(f"  {line}")
    if done.returncode != 0:
        print(f"  -> exit {done.returncode}")
    return done.returncode == 0


def run_checks() -> bool:
    """Both import orders, one runtime between them, and both libraries threaded.

    Reading the load commands says the patch is in the file; these say the
    loader agrees and neither runtime aborts when it starts.
    """
    orders = {
        "import torch, then finufft": "import torch\nimport finufft",
        "import finufft, then torch": "import finufft\nimport torch",
    }
    ok = True
    for name, imports in orders.items():
        done = subprocess.run(
            [sys.executable, "-c", LOADED.format(imports=imports)], capture_output=True, text=True
        )
        print(f"\n{name}")
        for line in (done.stdout + done.stderr).splitlines():
            print(f"  {line}")
        count = done.stdout.splitlines()[0].strip() if done.stdout.strip() else ""
        if done.returncode != 0 or count != "1":
            print(f"  -> expected one OpenMP runtime, exit 0; got {count or 'nothing'}, "
                  f"exit {done.returncode}")
            ok = False

    return _probe("threaded torch, then threaded finufft", WORKLOADS) and ok


# --- the three modes ----------------------------------------------------------


def describe(layout: Layout) -> Decision:
    torch_omp = read_macho(layout.torch_omp)
    finufft = read_macho(layout.finufft_lib)
    bundled = read_macho(layout.finufft_omp) if layout.finufft_omp else None

    print(f"torch    {layout.torch_omp}")
    print(f"         id {torch_omp.install_name}")
    print(f"finufft  {layout.finufft_lib}")
    for name, compat, current in finufft.loads:
        if _openmp_family(name):
            print(f"         loads {name}  (compat {compat}, current {current})")
    for rpath in finufft.rpaths:
        print(f"         rpath {rpath}")
    if bundled is not None:
        print(f"bundled  {layout.finufft_omp}")
        print(f"         id {bundled.install_name}")

    return decide(torch_omp, finufft, bundled, rpath_for(layout.finufft_lib, layout.torch_omp))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("diagnose", "patch", "verify"))
    ap.add_argument("--site", type=Path, help="look here instead of at the installed packages")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args(argv)

    if sys.platform != "darwin" and args.site is None:
        print("this is a macOS problem; nothing to do here", file=sys.stderr)
        return 0

    layout = find_layout(args.site)
    decision = describe(layout)
    print(f"\n{decision.action}: {decision.reason}")

    if args.mode == "diagnose":
        return 0 if decision.action in ("patch", "already") else 1

    if args.mode == "verify":
        if decision.action != "already":
            print("\nthe library is not pointed at torch's runtime", file=sys.stderr)
            return 1
        return 0 if run_checks() else 1

    if decision.action == "already":
        return 0
    if decision.action == "refuse":
        return 1

    print("\npatching:")
    apply(layout, decision, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
