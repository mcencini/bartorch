"""Sphinx configuration: pages in Markdown (MyST), the reference by autodoc over the package.

Importing ``bartorch`` needs torch but not the compiled library, so rendering
needs no native build.
"""

import os
import re
import sys
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / "src"))
# The generator of the API object index lives beside this file.
sys.path.insert(0, str(DOCS))

import api_objects  # noqa: E402
import colab  # noqa: E402

project = "bartorch"
author = "bartorch contributors"
copyright = "2024–2026, bartorch contributors"
#: Where the site is served from: GitHub Pages, from the gh-pages branch, with
#: one directory per published version.
PAGES_URL = "https://mcencini.github.io/bartorch"

#: The published version this build is, as the docs workflow names it:
#: ``latest`` for main and the tag for a release.  The version switcher marks
#: it, and the theme warns on a page whose version is not the newest release.
#: A build outside the workflow is ``latest``.
DOCS_RELEASE = os.environ.get("BARTORCH_DOCS_RELEASE", "latest")

#: The directory a reader should be sent to for this build's pages:
#: ``stable`` for the newest release, which is also archived under its tag,
#: and ``latest`` for main.  The canonical links point there.
DOCS_VERSION = os.environ.get("BARTORCH_DOCS_VERSION", "latest")

# The theme compares `release` with the version `versions.json` marks
# preferred to decide whether to warn that a page is not the current release,
# so a release build carries its tag, and `latest` warns.
version = release = DOCS_RELEASE

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
templates_path = ["_templates"]
# The gallery's section headers are sphinx-gallery's input, rendered into
# `auto_examples/`; `examples/index.md` beside them is a page of its own.
# The design notes are maintainers' records rather than documentation.
exclude_patterns = [
    "_build",
    "design",
    "examples/README.rst",
    "examples/*/README.rst",
    "Thumbs.db",
    ".DS_Store",
]

#: Set by ``scripts/build_docs_pdf.sh``.  The manual is built from ``manual.md``,
#: which places the landing page and the six sections side by side and adds
#: the page of every API object: the site reaches those from the API tables
#: rather than from the navigation, and a printed page has no links to follow.
PDF_MANUAL = os.environ.get("BARTORCH_DOCS_PDF") == "1"
if PDF_MANUAL:
    root_doc = "manual"
else:
    exclude_patterns.append("manual.md")

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
# References are footnotes, collected under each page's closing "References"
# heading; a rule above them would separate them from it.
myst_footnote_transition = False

# The stubs are generated from `api_objects.rst`, which `api_objects.write`
# produces from the API pages' tables before autosummary reads its sources.
# Named rather than True: True takes the pages to read from the environment
# of the previous build, which is empty on a clean checkout.
autosummary_generate = ["api_objects.rst"]
autodoc_member_order = "bysource"
# Types and defaults are stated in the docstrings' Parameters sections, which
# are the readable signature; the heading of an object's page is compact.
autodoc_typehints = "none"
autodoc_class_signature = "mixed"
autodoc_preserve_defaults = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_admonition_for_references = False

#: Objects a gallery example is about, rather than objects every example uses.
#: Only these pages carry a minigallery of the examples that use them.
GALLERY_BACKREFERENCES = {
    "bartorch.nufft",
    "bartorch.nufft_adjoint",
    "bartorch.linop.CartesianSense",
    "bartorch.linop.NoncartesianSense",
    "bartorch.linop.NUFFT",
    "bartorch.linop.LinearOperator",
    "bartorch.nlop.NonlinearSense",
    "bartorch.nlop.MultiEcho",
    "bartorch.nlop.IRGNM",
    "bartorch.optim.FISTA",
    "bartorch.optim.ADMM",
    "bartorch.optim.ADMMBlock",
    "bartorch.optim.data_scaling",
    "bartorch.priors.TotalVariation",
    "bartorch.priors.LocallyLowRank",
    "bartorch.priors.ImplicitPrior",
    "bartorch.learning.Denoiser",
    "bartorch.learning.Unrolled",
    "bartorch.tools.pics",
    "bartorch.tools.cc",
    "bartorch.tools.ecalib",
    "bartorch.tools.ncalib",
    "bartorch.tools.nlinv",
    "bartorch.tools.traj",
    "bartorch.tools.psf",
    "bartorch.tools.whiten",
}

autosummary_context = {
    "gallery_backreferences": sorted(GALLERY_BACKREFERENCES),
    "class_pages": api_objects.class_context(api_objects.collect(DOCS / "api")),
}

html_theme = "sphinx_book_theme"
html_title = "bartorch"
html_theme_options = {
    "repository_url": "https://github.com/mcencini/bartorch",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
    # The sidebar carries the six sections and the pages directly under
    # them.  Individual examples are reached from the section pages, and
    # individual objects from the tables on the API pages, whose stubs are
    # generated from `api_objects.rst` and so never enter this tree.
    "show_navbar_depth": 1,
    "max_navbar_depth": 2,
    "logo": {
        "image_light": "_static/bartorch-mark.svg",
        "image_dark": "_static/bartorch-mark-dark.svg",
        "alt_text": "bartorch",
    },
    # The list of published versions, written beside them by
    # scripts/publish_docs.py and fetched by the page when it loads, so a
    # build served from anywhere else shows no switcher.  The theme's check of
    # the list at build time is off: the list exists only once a version has
    # been published.
    "switcher": {
        "json_url": f"{PAGES_URL}/versions.json",
        "version_match": DOCS_RELEASE,
    },
    "check_switcher": False,
    "show_version_warning_banner": True,
    # The search field is in the sidebar; the header would otherwise carry a
    # second one on a wide screen.
    "navbar_persistent": [],
}
#: The theme's own sidebar, with the version switcher under the logo.
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "version-switcher.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}
html_baseurl = f"{PAGES_URL}/{DOCS_VERSION}/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/bartorch-mark.svg"

intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "torch": ("https://docs.pytorch.org/docs/stable", None),
    }
    if os.environ.get("BARTORCH_DOCS_ONLINE") == "1"
    else {}
)
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# -- Example gallery ---------------------------------------------------------

#: The gallery's sections, in the order a reader meets them.
GALLERY_SECTIONS = [
    "examples/01-basics",
    "examples/02-non-cartesian",
    "examples/03-applications",
    "examples/04-model-based",
    "examples/05-deep-learning",
]

#: Whether the examples are executed, which `./scripts/build_docs.sh --execute`
#: asks for.  Running them needs the compiled library, and several of them also
#: need `brainweb-dl`; a build that renders the pages from the scripts needs
#: neither, so it is what an ordinary documentation build does.
EXECUTE_EXAMPLES = os.environ.get("BARTORCH_DOCS_EXECUTE") == "1"

sphinx_gallery_conf = {
    "doc_module": "bartorch",
    "backreferences_dir": "generated/backreferences",
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    # Searched against each script's path; nothing matches the second.
    "filename_pattern": r".*\.py" if EXECUTE_EXAMPLES else r"(?!)",
    "nested_sections": True,
    "subsection_order": ExplicitOrder(GALLERY_SECTIONS),
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    # Left off deliberately: it strips `# sphinx_gallery_start_ignore` and its
    # partner along with the rest of the in-file configuration comments, and
    # the pass below needs those markers to know what to keep off the page.
    "remove_config_comments": False,
}


def _hide_ignored_code_from_the_page_only() -> None:
    """Keep the page free of the blocks an example hides, and nothing else.

    sphinx-gallery strips its ignore blocks once, before it writes either the
    page or the notebook, so a downloaded notebook is missing whatever the page
    hides and raises on the first cell that needed it.  Stripping them as the
    page is written instead leaves the downloadable script and notebook whole.

    A cell that is hidden in full renders as nothing rather than as an empty
    ``code-block`` directive.  Its output -- the figures it drew, what it
    printed -- is emitted separately and is kept either way.
    """
    from sphinx_gallery import gen_rst, py_source_parser

    strip = py_source_parser.remove_ignore_blocks

    def keep(code):
        strip(code)  # for its check that every flag has its partner
        return code

    py_source_parser.remove_ignore_blocks = keep

    original = gen_rst.codestr2rst

    def codestr2rst(code, *args, **kwargs):
        shown = strip(code)
        return original(shown, *args, **kwargs) if shown.strip() else ""

    gen_rst.codestr2rst = codestr2rst

    write_notebook = gen_rst.jupyter_notebook

    def jupyter_notebook(script_blocks, *args, **kwargs):
        """The notebook keeps the code, but not the flags that hid it."""
        return write_notebook(
            [block._replace(content=_unflagged(block.content)) for block in script_blocks],
            *args,
            **kwargs,
        )

    gen_rst.jupyter_notebook = jupyter_notebook


def _unflagged(content: str) -> str:
    """The block without the comment lines that mark a hidden region."""
    return "\n".join(
        line
        for line in content.splitlines()
        if line.strip() not in ("# sphinx_gallery_start_ignore", "# sphinx_gallery_end_ignore")
    )


def _compact_signature(_app, what, _name, _obj, _options, _signature, return_annotation):
    """Head an object's page with ``Name()`` rather than its whole signature.

    The Parameters section below states every argument with its type and
    default, which is the readable form of the signature; ``inspect.signature``
    is unchanged.  Properties and data keep what autodoc gives them.
    """
    if what in {"function", "method", "class", "exception"}:
        return "()", return_annotation
    return None


def _public_bases(_app, _name, _obj, _options, bases):
    """Name a private base class by the nearest public class it is built on.

    A private class has no page to link to, and ``object`` says nothing.
    """
    kept = []
    for base in bases:
        if isinstance(base, type):
            base = next(
                (klass for klass in base.__mro__ if not klass.__name__.startswith("_")), object
            )
        if base is not object:
            kept.append(base)
    bases[:] = kept


#: The README is the repository's front page and the documentation's.  On
#: GitHub and PyPI its figures are fetched from ``main`` and its links point at
#: a published version of the site; here they become this build's static files
#: and pages, so each version's landing page links within that version.
_SITE_PAGE = re.compile(
    r"https://mcencini\.github\.io/bartorch/(?:latest|stable|v\d+\.\d+\.\d+)/"
    r"([^\s)\"'<>#]+)\.html"
)
_RAW_STATIC = re.compile(r"https://raw\.githubusercontent\.com/mcencini/bartorch/main/docs/_static/")
_PICTURE = re.compile(
    r"<picture>\s*<source[^>]*srcset=\"(?P<dark>[^\"]+)\"[^>]*>\s*"
    r"<img (?P<attributes>[^>]*)src=\"(?P<light>[^\"]+)\"(?P<rest>[^>]*)>\s*</picture>"
)


def _readme_for_docs(text: str) -> str:
    """The README with its figures and links resolved within this build.

    A ``<picture>`` that chooses a variant by the reader's colour scheme
    becomes the two images the theme shows one of, since the theme's own
    light and dark switch does not reach a media query.
    """
    text = _RAW_STATIC.sub("_static/", text)
    text = _SITE_PAGE.sub(r"\1.md", text)
    return _PICTURE.sub(
        r'<img class="only-light" \g<attributes>src="\g<light>"\g<rest>>'
        r'<img class="only-dark" \g<attributes>src="\g<dark>"\g<rest>>',
        text,
    )


def _included_readme(_app, relative_path, parent_docname, content):
    """Apply :func:`_readme_for_docs` to the README the landing page includes."""
    if parent_docname == "index" and Path(relative_path).name == "README.md":
        content[0] = _readme_for_docs(content[0])


_TOCTREE = re.compile(r"```\{toctree\}.*?```\n?", re.S)


def _landing_page_in_the_manual(_app, docname, source):
    """The landing page without its toctree, when ``manual.md`` holds the sections."""
    if PDF_MANUAL and docname == "index":
        source[0] = _TOCTREE.sub("", source[0])


def _object_pages_in_the_manual(_app, doctree) -> None:
    """Unwrap autosummary's toctrees, for the PDF only.

    The single-page builder inlines each object's page where the toctree
    naming it stands, and the HTML writer skips everything inside the node
    autosummary wraps its toctree in.
    """
    if not PDF_MANUAL:
        return
    from sphinx.ext.autosummary import autosummary_toc

    for node in list(doctree.findall(autosummary_toc)):
        node.replace_self(node.children)


def _colab_badge(_app, docname, source) -> None:
    """Put the Open in Colab badge under an example page's title."""
    source[0] = colab.with_badge(source[0], docname, DOCS_RELEASE)


def _colab_notebooks(app, exception) -> None:
    """Write the Colab copies of the gallery's notebooks into the built site."""
    if exception is None and app.builder.name == "html" and not PDF_MANUAL:
        colab.write(Path(app.srcdir), Path(app.outdir), DOCS_RELEASE)


def _write_api_object_index(app) -> None:
    """Write the page the API stubs are generated from, ahead of autosummary."""
    api_objects.write(app.srcdir)


def setup(app):
    """Wire in the passes this configuration adds."""
    _hide_ignored_code_from_the_page_only()
    app.connect("autodoc-process-signature", _compact_signature)
    app.connect("autodoc-process-bases", _public_bases)
    app.connect("include-read", _included_readme)
    app.connect("source-read", _landing_page_in_the_manual)
    app.connect("doctree-read", _object_pages_in_the_manual)
    app.connect("source-read", _colab_badge)
    app.connect("build-finished", _colab_notebooks)
    # Ahead of autosummary's own handler, which reads the sources it writes
    # stubs for: a page written after it would be read a build late.
    app.connect("builder-inited", _write_api_object_index, priority=100)
