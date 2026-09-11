"""Sphinx configuration: pages in Markdown (MyST), the reference by autodoc over the package.

Importing ``bartorch`` needs torch but not the compiled library, so rendering
needs no native build; executing the gallery does.
"""

import os
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS / "_ext"))
sys.path.insert(0, str(ROOT / "src"))

project = "bartorch"
author = "bartorch contributors"
copyright = "2024–2026, bartorch contributors"
release = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)[1]
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
exclude_patterns = ["_build", "_ext", "gallery_src", "examples", "Thumbs.db", ".DS_Store"]

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

# The gallery is rendered without running unless asked; a failing example then
# fails the build.
plot_gallery = "True" if os.environ.get("BARTORCH_DOCS_EXECUTE") == "1" else "False"
sphinx_gallery_conf = {
    "examples_dirs": str(DOCS / "gallery_src"),
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"__init__\.py",
    "within_subsection_order": "gallery_order.ExampleOrder",
    "nested_sections": True,
    "download_all_examples": True,
    "backreferences_dir": None,
    "doc_module": (),
    "reference_url": {},
    "image_scrapers": ("matplotlib",),
    "abort_on_example_error": True,
    "only_warn_on_example_error": False,
    "remove_config_comments": True,
}
