"""Sphinx configuration: pages in Markdown (MyST), the reference by autodoc over the package.

Importing ``bartorch`` needs torch but not the compiled library, so rendering
needs no native build.
"""

import importlib.metadata
import os
import sys
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / "src"))

project = "bartorch"
author = "bartorch contributors"
copyright = "2024–2026, bartorch contributors"
# From the installed package, because the version is the git tag now and
# `pyproject.toml` no longer carries it.  Rendering from a checkout with
# nothing installed is the ordinary case here, so it is not an error.
try:
    release = importlib.metadata.version("bartorch")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0.dev0"
version = release

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
exclude_patterns = ["_build", "design", "examples", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_member_order = "bysource"
# Types are stated in the docstrings; the signature stays readable.
autodoc_typehints = "none"
autodoc_class_signature = "mixed"
autodoc_preserve_defaults = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

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
    "show_navbar_depth": 2,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

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
    "backreferences_dir": "api/generated/backreferences",
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


def setup(app):
    """Wire in the passes this configuration adds."""
    _hide_ignored_code_from_the_page_only()
