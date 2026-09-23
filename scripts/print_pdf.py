"""Print the single-page HTML build of the documentation to one PDF.

    python scripts/print_pdf.py docs/_build/singlehtml docs/_build/bartorch-docs.pdf

Headless Chromium loads the page, waits for MathJax to typeset every equation,
and prints it, so the manual shows the equations, figures and tables the site
shows.  A cover page with the six sections is put in front, the navigation,
the dark-mode variants of the figures and the download buttons are left out,
and the PDF carries an outline built from the headings.

The page loads MathJax from the CDN Sphinx names.  Where that host cannot be
reached, ``BARTORCH_PDF_MATHJAX`` names a local copy of MathJax 3's ``es5``
directory to serve it from instead.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path

#: The six sections, by the document their landing page is, in reading order.
SECTIONS = (
    ("User guide", "guides/user/index"),
    ("Developer guide", "guides/developer/index"),
    ("Explanation", "explanation/index"),
    ("Examples", "examples/index"),
    ("API reference", "api/index"),
    ("Misc", "misc/index"),
    ("API objects", "api_objects"),
)

#: What a printed page does not need: the theme's navigation, the figures meant
#: for a dark background, and the controls that only work in a browser.
PRINT_CSS = """
#jb-print-docs-body, section#bartorch-documentation > h1, #bd-header-version-warning,
.version-switcher__container,
.bd-header, .bd-sidebar-primary, .bd-sidebar-secondary, .header-article,
.bd-footer, .bd-footer-article, .bd-footer-content, .prev-next-area,
.skip-link, .pst-async-banner-revealer, #pst-scroll-pixel-helper,
.only-dark, .copybtn, .headerlink,
.sphx-glr-download-link-note, .sphx-glr-footer, .sphx-glr-signature,
.sphx-glr-download, .sphx-glr-script-out .copybtn, .sphx-glr-timing {
  display: none !important;
}
html, body { background: #ffffff !important; }
.bd-main, .bd-content, .bd-article-container, .bd-article {
  max-width: none !important; width: auto !important; margin: 0 !important;
  padding: 0 !important;
}
body { font-size: 10pt; }
pre, table, figure, .sphx-glr-thumbcontainer, img { break-inside: avoid; }
h1, h2, h3 { break-after: avoid; }
img { max-width: 100% !important; height: auto; }
pre { white-space: pre-wrap !important; overflow-wrap: anywhere; }
.pdf-cover { break-after: page; padding-top: 30mm; }
.pdf-cover img { width: 110mm; margin-bottom: 12mm; }
.pdf-cover h1 { font-size: 24pt; margin: 0 0 4mm; border: none; }
.pdf-cover p { color: #475467; margin: 0 0 10mm; }
.pdf-cover li { margin: 0.4em 0; font-size: 12pt; }
"""

FOOTER = (
    '<div style="width: 100%; font-size: 8px; color: #667085; text-align: center;">'
    '<span class="pageNumber"></span></div>'
)


def _cover(version: str) -> str:
    items = "".join(
        f'<li><a href="#document-{target}">{label}</a></li>' for label, target in SECTIONS
    )
    return (
        '<section class="pdf-cover">'
        '<img src="_static/bartorch-logo.svg" alt="bartorch">'
        "<h1>bartorch documentation</h1>"
        f"<p>{version}</p>"
        f"<h2>Contents</h2><ul>{items}</ul>"
        "</section>"
    )


def _serve_mathjax(directory: Path):
    """A route handler answering MathJax's requests from a local ``es5`` copy."""

    def handle(route):
        path = route.request.url.split("/es5/", 1)[-1].split("?", 1)[0]
        local = directory / path
        if "/es5/" in route.request.url and local.is_file():
            kind = mimetypes.guess_type(local.name)[0] or "application/octet-stream"
            route.fulfill(path=str(local), content_type=kind)
        else:
            route.abort()

    return handle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", type=Path, help="the singlehtml build directory")
    parser.add_argument("output", type=Path, help="the PDF to write")
    parser.add_argument("--version", default="", help="the version printed on the cover")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for MathJax")
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "print_pdf.py: Playwright is not installed:\n"
            "  pip install playwright && python -m playwright install chromium",
            file=sys.stderr,
        )
        return 1

    # The single-page builder names its page after the root document.
    page_file = (args.site / "manual.html").resolve()
    with sync_playwright() as playwright:
        executable = os.environ.get("BARTORCH_PDF_CHROMIUM") or None
        browser = playwright.chromium.launch(executable_path=executable)
        page = browser.new_page()
        local = os.environ.get("BARTORCH_PDF_MATHJAX")
        if local:
            page.route("**/mathjax@3/**", _serve_mathjax(Path(local)))
        page.goto(page_file.as_uri(), wait_until="load", timeout=args.timeout * 1000)
        page.wait_for_function(
            "() => window.MathJax && window.MathJax.startup && window.MathJax.startup.promise",
            timeout=args.timeout * 1000,
        )
        page.evaluate("() => window.MathJax.startup.promise")
        untypeset = page.evaluate(
            "() => document.querySelectorAll('.math:not(:has(mjx-container))').length"
        )
        if untypeset:
            print(f"print_pdf.py: {untypeset} equations were not typeset", file=sys.stderr)
            return 1
        page.add_style_tag(content=PRINT_CSS)
        page.evaluate(
            """([cover, documents]) => {
                const main = document.querySelector('.bd-article') || document.body;
                main.insertAdjacentHTML('afterbegin', cover);
                for (const image of document.querySelectorAll('img')) {
                    if (/^https?:/.test(image.getAttribute('src') || '')) image.remove();
                }
                // Each section starts a page; the objects inside one do not.
                for (const name of documents) {
                    const anchor = document.getElementById('document-' + name);
                    const start = anchor && anchor.nextElementSibling;
                    if (start) start.style.breakBefore = 'page';
                }
            }""",
            [_cover(args.version), ["index", *(target for _, target in SECTIONS)]],
        )
        page.emulate_media(media="print", color_scheme="light")
        page.wait_for_load_state("networkidle")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        page.pdf(
            path=str(args.output),
            format="A4",
            margin={"top": "16mm", "bottom": "18mm", "left": "15mm", "right": "15mm"},
            print_background=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=FOOTER,
            outline=True,
            tagged=True,
        )
        browser.close()
    print(f"print_pdf.py: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
