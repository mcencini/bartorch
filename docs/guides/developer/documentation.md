# Documentation

## Building

From the repository root, in a Python environment with PyTorch (a CPU build is
enough):

```bash
python -m pip install -r docs/requirements.txt
./scripts/build_docs.sh
```

Open `docs/_build/html/index.html`.  The build imports `bartorch` from `src/`,
which needs torch but not the compiled library, and renders the gallery
without running it.  Warnings are errors, as in the documentation workflow.

To run the gallery, build the library first (see {doc}`toolchain`), then:

```bash
./scripts/build_docs.sh --execute
```

A failing example fails the build.  Examples with the `optional_` prefix are
rendered but not run; to run them too, install `docs/requirements-examples.txt`
and pass
`-D sphinx_gallery_conf.filename_pattern='/(plot_|optional_)'` to Sphinx.
`--online` fetches intersphinx inventories.  Check external links with
`python -m sphinx -b linkcheck docs docs/_build/linkcheck`.

## Pages and the reference

Pages are Markdown, parsed by MyST; a directive is written as a fenced block,
and reStructuredText only inside `{eval-rst}`.  Each page of the API reference
(`docs/api/`) is a set of sections, each an `autosummary` table of the names it
documents; autosummary writes a page per name into `docs/api/generated/`, and
autodoc renders its docstring.  `tests/test_docs.py` fails when a public name is
missing from the reference or the reference lists one that is not public.

A command without a hand-written wrapper is built at import from BART's
catalogue, with BART's help as its docstring; its reference page is rendered
from that docstring like any other.

## Gallery examples

Place a standalone script in a numbered subsection of `docs/gallery_src/`: `plot_`
for core CPU examples, `optional_` for ones needing optional libraries.  Begin
with a module docstring holding a title, the objective, prerequisites and
scope; separate narrative and code with `# %%` blocks.  A new subsection needs
a `GALLERY_HEADER.md`.

Use fixed seeds, small synthetic inputs and visible results, and include a
check against something outside BART, such as a closed form or an adjoint
identity.  No downloads, hidden datasets, blanket exception handling or
installation commands.

## Docstrings

Docstrings are NumPy style and written for a developer reading the code.

- State what is not obvious from the name, the signature and the annotations:
  units, coordinate frames, composition order, invariants, side effects,
  preconditions, special return values, and behaviour a maintainer could break
  by accident.
- Do not restate the signature, narrate the implementation, or document a
  parameter with its own name.  List a parameter only when its entry adds
  something.
- A module docstring states the module's responsibility in a sentence or a few.
  Design rationale belongs in `AGENTS.md` or in a comment beside the code it
  explains.
- A private helper gets a docstring only for a contract that is not obvious.
- Precision over brevity: a short paragraph for a real invariant is better than
  an ambiguous line.
