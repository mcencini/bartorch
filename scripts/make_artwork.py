"""Write the bartorch logo, its compact mark and two explanation figures into ``docs/_static``.

    python scripts/make_artwork.py

The logo is BART's own mark followed by the PyTorch flame and the word
``torch``.

* The BART mark -- the letters and the two corner brackets -- is read from
  ``external/bart/src/geom/logo.c``, the outline ``bart phantom -B`` draws.
  That file is part of BART's source and is distributed under BART's
  BSD-3-Clause license (``external/bart/LICENSE``), whose notice travels with
  it in ``docs/misc/license.md``.  Each segment there is a cubic Hermite spline
  per coordinate, converted to a Bezier curve with the same matrix BART's
  ``cspline2bezier`` uses.
* The flame is the PyTorch logo, reproduced unaltered in shape and colour
  from ``assets/images/logo-icon.svg`` of github.com/pytorch/pytorch.github.io
  and only scaled.  PyTorch, the PyTorch logo and any related marks are
  trademarks of The Linux Foundation; ``docs/misc/license.md`` attributes it.
* ``torch`` is set in DejaVu Sans Bold, the font matplotlib ships, compressed
  horizontally and converted to outlines, so the SVG needs no font.  It is not
  the PyTorch wordmark.

Nothing here implies endorsement by the BART developers or the PyTorch
Foundation; ``docs/misc/license.md`` says so where the logo is shown.

Four images are written, each in a light and a dark variant: the horizontal
logo, the compact mark -- BART's brackets around the flame -- used in the sidebar
and as the favicon, the architecture figure the README and
``docs/explanation/execution-model.md`` show, and the encoding-form figure of
``docs/explanation/encoding.md``.  The figures' text is SVG text in the
reader's sans-serif font; no font is embedded.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "external" / "bart" / "src" / "geom" / "logo.c"
OUT = ROOT / "docs" / "_static"

#: How many segments each closed outline of ``logo.c`` has, in file order:
#: B, its two counters, A, its counter, R, its counter, T, and the brackets.
#: The grouping is ``src/simu/phantom.c``'s.
CONTOURS = (11, 6, 6, 8, 4, 16, 6, 8, 6, 6)

#: Ink for a light background and for a dark one.
INK = {"light": "#1f2933", "dark": "#e4e7eb"}
ACCENT = {"light": "#2f6f9f", "dark": "#7fb3dc"}

#: The wordmark's horizontal compression, towards the proportions of BART's
#: condensed letters.
CONDENSE = 0.86


def _segments() -> list[tuple[list[float], list[float]]]:
    """``logo.c``'s segments as ``(vertical, horizontal)`` Hermite coefficients."""
    body = SOURCE.read_text().split("=", 1)[1]
    values = [float(v) for v in re.findall(r"-?\d+\.\d+", body)]
    return [(values[i : i + 4], values[i + 4 : i + 8]) for i in range(0, len(values), 8)]


def _bezier(hermite: list[float]) -> list[float]:
    """BART's ``cspline2bezier``: ``(p0, m0, p1, m1)`` to four control points."""
    p0, m0, p1, m1 = hermite
    return [p0, p0 + m0 / 3.0, p1 - m1 / 3.0, p1]


def _outlines(first: int, last: int) -> str:
    """SVG path data of outlines ``first`` to ``last - 1``, in ``logo.c``'s units."""
    segments = _segments()
    start = sum(CONTOURS[:first])
    commands = []
    for count in CONTOURS[first:last]:
        for index, (vertical, horizontal) in enumerate(segments[start : start + count]):
            y, x = _bezier(vertical), _bezier(horizontal)
            if index == 0:
                commands.append(f"M{x[0]:.2f} {y[0]:.2f}")
            commands.append(f"C{x[1]:.2f} {y[1]:.2f} {x[2]:.2f} {y[2]:.2f} {x[3]:.2f} {y[3]:.2f}")
        commands.append("Z")
        start += count
    return " ".join(commands)


def bart_mark() -> str:
    """The whole of BART's mark: the four letters and the two brackets."""
    return _outlines(0, len(CONTOURS))


def word(text: str, height: float, x: float, baseline: float) -> tuple[str, float]:
    """Outlines of ``text`` with ascenders ``height`` tall, and the right edge they reach."""
    font = FontProperties(
        fname=str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
    )
    probe = TextPath((0, 0), "h", size=1.0, prop=font)
    size = height / probe.get_extents().y1
    outline = TextPath((0, 0), text, size=size, prop=font)
    left = outline.get_extents().x0
    commands = []
    points = []
    for vertices, code in outline.iter_segments(curves=True, simplify=False):
        xy = [
            (x + CONDENSE * (vertices[i] - left), baseline - vertices[i + 1])
            for i in range(0, len(vertices), 2)
        ]
        points += xy
        coordinates = " ".join(f"{a:.2f} {b:.2f}" for a, b in xy)
        if code == MplPath.MOVETO:
            commands.append(f"M{coordinates}")
        elif code == MplPath.LINETO:
            commands.append(f"L{coordinates}")
        elif code == MplPath.CURVE3:
            commands.append(f"Q{coordinates}")
        elif code == MplPath.CURVE4:
            commands.append(f"C{coordinates}")
        elif code == MplPath.CLOSEPOLY:
            commands.append("Z")
    right = max(a for a, _ in points)
    return " ".join(commands), right


#: The PyTorch logo's flame, as ``logo-icon.svg`` draws it: its path, the
#: circle beside it, their colour and the box they are drawn in.
FLAME_PATH = (
    "M77.6,1099.6l-8.1,8.1c13.3,13.3,13.3,34.7,0,47.8c-13.3,13.3-34.7,13.3-47.8,0"
    "c-13.3-13.3-13.3-34.7,0-47.8l0,0l21.1-21.1l3-3l0,0v-15.9l-31.8,31.8"
    "c-17.7,17.7-17.7,46.3,0,64c17.7,17.7,46.3,17.7,63.7,0"
    "C95.3,1145.8,95.3,1117.4,77.6,1099.6z"
)
FLAME_CIRCLE = (61.7, 1091.8, 5.9)
FLAME_COLOUR = "#EE4C2C"
FLAME_BOX = (0.6, 1067.9, 90.3, 109.1)


def flame(x: float, top: float, height: float) -> tuple[str, float]:
    """The flame scaled to ``height`` with its box's corner at ``(x, top)``; and its right edge."""
    left, upper, width, tall = FLAME_BOX
    scale = height / tall
    cx, cy, r = FLAME_CIRCLE
    element = (
        f'  <g transform="translate({x:.2f} {top:.2f}) scale({scale:.5f}) '
        f'translate({-left} {-upper})" fill="{FLAME_COLOUR}">\n'
        f'    <path d="{FLAME_PATH}"/>\n'
        f'    <circle cx="{cx}" cy="{cy}" r="{r}"/>\n'
        "  </g>"
    )
    return element, x + width * scale


def _svg(
    width: float,
    height: float,
    paths: list[tuple[str, str]],
    label: str,
    margin=8.0,
    extra: tuple[str, ...] = (),
) -> str:
    body = "\n".join(
        [f'  <path fill="{colour}" fill-rule="evenodd" d="{data}"/>' for colour, data in paths]
        + list(extra)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}" '
        f'viewBox="{-margin:.0f} {-margin:.0f} '
        f'{width + 2 * margin:.0f} {height + 2 * margin:.0f}">\n'
        f"  <title>{label}</title>\n{body}\n</svg>\n"
    )


#: BART's cap height and baseline in ``logo.c``'s units, read off the letters.
CAP, BASELINE = 136.4, 180.0


def logo(theme: str) -> str:
    """The horizontal logo: BART's mark, the PyTorch flame, then ``torch``."""
    # The flame stands on BART's baseline and reaches a little above its
    # capitals, as the dot above it rises past a letter.
    height = 1.08 * CAP
    element, right = flame(398.0, BASELINE - height, height)
    # Lowercase letters of this font look larger than BART's condensed capitals
    # at the same height, so the ascenders stop a little short of the caps.
    letters, right = word("torch", 0.93 * CAP, right + 12.0, BASELINE)
    width = right + 4.0
    return _svg(
        width,
        224.0,
        [(INK[theme], bart_mark()), (ACCENT[theme], letters)],
        "bartorch",
        extra=(element,),
    )


def mark(theme: str) -> str:
    """The compact mark: BART's two brackets around the PyTorch flame.

    The brackets keep their shapes; the lower one is moved in so that the
    pair frames a square, which is what a favicon has room for.
    """
    # In logo.c the upper bracket spans x 8.24..82.8 and y 6.0..90.9, the lower
    # one x 300.3..374.9 and y 135.4..220.2.
    side = 220.2 - 6.0
    shift = 374.88 - (8.24 + side)
    height = 0.72 * side
    width = height * FLAME_BOX[2] / FLAME_BOX[3]
    element, _ = flame(8.24 + (side - width) / 2.0, 6.0 + (side - height) / 2.0, height)
    lower = (
        f'  <path fill="{INK[theme]}" fill-rule="evenodd" transform="translate({-shift:.2f} 0)" '
        f'd="{_outlines(len(CONTOURS) - 1, len(CONTOURS))}"/>'
    )
    return _svg(
        side + 2 * 8.24,
        224.0,
        [(INK[theme], _outlines(len(CONTOURS) - 2, len(CONTOURS) - 1))],
        "bartorch",
        extra=(lower, element),
    )


#: Box fill, box outline and arrow colour of the architecture figure.
PANEL = {"light": "#eef4f9", "dark": "#16222d"}
EDGE = {"light": "#2f6f9f", "dark": "#7fb3dc"}
MUTED = {"light": "#52606d", "dark": "#9aa5b1"}

FONT = "DejaVu Sans, Verdana, Helvetica, Arial, sans-serif"


def _box(x, y, width, height, theme, title, lines=(), anchor="middle"):
    """A rounded box with a bold title and lines under it."""
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" '
        f'fill="{PANEL[theme]}" stroke="{EDGE[theme]}" stroke-width="1.5"/>'
    ]
    tx = x + width / 2 if anchor == "middle" else x + 16
    rows = [(title, "bold", INK[theme])] + [(line, "normal", MUTED[theme]) for line in lines]
    top = y + height / 2 - (len(rows) - 1) * 11 + 5
    for index, (text, weight, colour) in enumerate(rows):
        parts.append(
            f'<text x="{tx}" y="{top + 22 * index:.0f}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{colour}">{text}</text>'
        )
    return "\n  ".join(parts)


def _arrow(x, y0, y1, theme):
    return (
        f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1 - 7}" stroke="{EDGE[theme]}" stroke-width="2"/>'
        f'<path d="M{x - 6} {y1 - 9} L{x} {y1} L{x + 6} {y1 - 9} Z" fill="{EDGE[theme]}"/>'
    )


def architecture(theme: str) -> str:
    """Tensors, the two kinds of interface, the ABI, BART, and what serves BART."""
    width, gap = 760, 22
    left, right = 0, width / 2 + 10
    half = width / 2 - 10
    rows = [
        _box(0, 0, width, 44, theme, "PyTorch tensors, on the CPU or a CUDA device"),
        _box(
            left,
            66,
            half,
            108,
            theme,
            "Command-style interface",
            (
                "bartorch.tools: BART commands",
                "bartorch.fft, fwt, ...: array functions",
                "bartorch CLI: the bart command line",
            ),
        ),
        _box(
            right,
            66,
            half,
            108,
            theme,
            "Composable interface",
            (
                "linop, nlop: linear and nonlinear operators",
                "optim, priors: solvers, regularizers",
                "apps, learning, interop: pipelines, adapters",
            ),
        ),
        _box(0, 196, width, 44, theme, "ctypes  \u2192  bartorch C ABI (libbartorch)"),
        _box(
            0,
            262,
            width,
            66,
            theme,
            "Embedded BART",
            ("commands, linear and nonlinear operators, iterative algorithms",),
        ),
        _box(
            0,
            350,
            width,
            88,
            theme,
            "Substituted backends",
            (
                "FINUFFT, cuFINUFFT: non-uniform Fourier transforms",
                "BLAS, LAPACK: MKL, PyTorch or SciPy  \u00b7  FFT: MKL or pocketfft",
            ),
        ),
    ]
    arrows = [
        _arrow(half / 2, 44, 66, theme),
        _arrow(right + half / 2, 44, 66, theme),
        _arrow(half / 2, 174, 196, theme),
        _arrow(right + half / 2, 174, 196, theme),
        _arrow(width / 2, 240, 262, theme),
        _arrow(width / 2, 328, 350, theme),
    ]
    del gap
    body = "\n  ".join(rows + arrows)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="bartorch architecture" viewBox="-2 -2 764 442" '
        f'font-family="{FONT}" font-size="15">\n'
        "  <title>bartorch architecture</title>\n"
        f"  {body}\n</svg>\n"
    )


def _harrow(x0, x1, y, theme):
    return (
        f'<line x1="{x0}" y1="{y}" x2="{x1 - 7}" y2="{y}" stroke="{EDGE[theme]}" stroke-width="2"/>'
        f'<path d="M{x1 - 9} {y - 6} L{x1} {y} L{x1 - 9} {y + 6} Z" fill="{EDGE[theme]}"/>'
    )


def encoding(theme: str) -> str:
    """The forward path of the encoding form, from coefficient images to samples."""
    steps = [
        ("x[a]", ("image or", "coefficients")),
        ("× I[c, a, t]", ("sensitivities,", "spatial weights")),
        ("Tₜ", ("FFT, NUFFT", "or wave")),
        ("× O[a, t]", ("pattern, basis,", "density weights")),
        ("∑ₐ", ("contraction", "over terms")),
        ("y[c, t]", ("samples", "")),
    ]
    box, gap, height = 128, 20, 92
    width = len(steps) * box + (len(steps) - 1) * gap
    parts = []
    for index, (title, lines) in enumerate(steps):
        x = index * (box + gap)
        parts.append(_box(x, 0, box, height, theme, title, tuple(line for line in lines if line)))
        if index:
            parts.append(_harrow(x - gap, x, height / 2, theme))
    body = "\n  ".join(parts)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="The encoding form" viewBox="-2 -2 {width + 4} {height + 4}" '
        f'font-family="{FONT}" font-size="14">\n'
        "  <title>The encoding form</title>\n"
        f"  {body}\n</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "-dark"
        (OUT / f"bartorch-logo{suffix}.svg").write_text(logo(theme))
        (OUT / f"bartorch-mark{suffix}.svg").write_text(mark(theme))
        (OUT / f"architecture{suffix}.svg").write_text(architecture(theme))
        (OUT / f"encoding{suffix}.svg").write_text(encoding(theme))
    print(f"make_artwork.py: wrote the logo, the mark and the two figures into {OUT}")


if __name__ == "__main__":
    main()
