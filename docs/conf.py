import os
import sys

# Repository root is one level above docs/.  The package lives under src/ in the
# modern src-layout; add that so Sphinx autodoc can import bartorch directly.
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))

project = "bartorch"
author = "bartorch contributors"
version = "0.1.0"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/mcencini/bartpy",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
}

# Enable intersphinx cross-references when internet access is available
# (ReadTheDocs builds).  Disable in offline CI environments to avoid network
# warnings that would break a -W sphinx-build invocation.
_online = os.environ.get("READTHEDOCS") == "True"
intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "torch": ("https://pytorch.org/docs/stable", None),
    }
    if _online
    else {}
)

# Notebooks live in the top-level examples/ directory (symlinked as docs/examples/)
# and are executed at build time so the rendered pages carry their outputs.
nbsphinx_execute = "always"
# A cell that raises renders its traceback instead of failing the build.
nbsphinx_allow_errors = True

# Raw-markdown cells with fenced code confuse the IPython lexer, and a figure
# whose cell raised mid-way has no file behind it.
suppress_warnings = [
    "misc.highlighting_failure",
    "image.not_readable",
]

# Follow symlinks so that docs/examples/ → ../examples/ is resolved correctly.
# This is the default in Sphinx ≥ 7 but we set it explicitly for clarity.
html_extra_path = []

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]
html_static_path = ["_static"]
