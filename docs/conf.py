"""Build the reference without importing torch or loading libbartorch."""
import os
import re
import sys
from pathlib import Path

from sphinx_gallery.sorting import FileNameSortKey

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
    "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx", "sphinx_copybutton", "sphinx_gallery.gen_gallery",
    "bartorch_api",
]
html_theme = "sphinx_book_theme"
html_title = "bartorch"
html_theme_options = {
    "repository_url": "https://github.com/mcencini/bartpy",
    "repository_branch": "main", "path_to_docs": "docs",
    "use_repository_button": True, "use_issues_button": True,
    "use_edit_page_button": True, "show_navbar_depth": 2,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
exclude_patterns = [
    "_build", "_ext", "gallery_src", "examples", "README.rst", "Thumbs.db", ".DS_Store",
]
intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "torch": ("https://docs.pytorch.org/docs/stable", None),
    }
    if os.environ.get("BARTORCH_DOCS_ONLINE") == "1" else {}
)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = "description"
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
# Rendering alone needs no native build; explicit execution fails on errors.
plot_gallery = os.environ.get("BARTORCH_DOCS_EXECUTE") == "1"
sphinx_gallery_conf = {
    "examples_dirs": str(DOCS / "gallery_src"),
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"__init__\.py",
    "within_subsection_order": FileNameSortKey,
    "subsection_order": sorted,
    "nested_sections": True,
    "download_all_examples": True,
    "backreferences_dir": None,
    "doc_module": (), "reference_url": {},
    "image_scrapers": ("matplotlib",),
    "abort_on_example_error": True,
    "only_warn_on_example_error": False,
    "remove_config_comments": True,
}
