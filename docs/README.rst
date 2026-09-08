Building the documentation
==========================

From the repository root, in a Python 3.10+ virtual environment::

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs docs/_build/html

Open ``docs/_build/html/index.html``. The default build reads Python source
statically and renders gallery scripts without executing them. It requires
neither PyTorch nor a compiled BART library. Generated reference/gallery files
are ignored by Git. Documentation source changes are confined to ``docs/``.

To execute the core gallery, first install your chosen PyTorch build and a
working bartorch development installation (see the Developer Guide), then::

   python -m pip install -r docs/requirements.txt
   BARTORCH_DOCS_EXECUTE=1 python -m sphinx -b html -W --keep-going \
       -D sphinx_gallery_conf.run_stale_examples=True docs docs/_build/executed

Optional examples use the ``optional_`` filename prefix and are rendered but
excluded from core execution. Install ``docs/requirements-examples.txt`` and
run all examples with::

   BARTORCH_DOCS_EXECUTE=1 python -m sphinx -b html -W --keep-going \
       -D sphinx_gallery_conf.filename_pattern='/(plot_|optional_)' \
       -D sphinx_gallery_conf.run_stale_examples=True docs docs/_build/full

Examples never download datasets or pretrained weights. They use small CPU
phantoms. Execution failures fail the build. A rendering-only build is not
validation of numerical examples.

Set ``BARTORCH_DOCS_ONLINE=1`` to fetch intersphinx inventories. Check external
links separately with::

   python -m sphinx -b linkcheck docs docs/_build/linkcheck

The package's existing ``docs`` extra describes the earlier notebook setup.
Use these requirements until packaging changes are coordinated with core
work. The legacy ``docs/examples`` symlink is excluded, not modified.
Hosting configuration is left to the repository owner.
