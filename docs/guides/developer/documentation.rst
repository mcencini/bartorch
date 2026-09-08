Writing and building documentation
==================================

.. include:: ../../README.rst
   :start-after: ==========================

Adding a gallery example
------------------------

Place a standalone script in a numbered subsection of ``docs/gallery_src/``.
Use ``plot_`` for core CPU examples and ``optional_`` for optional libraries.
Begin with a module docstring containing an RST title, objective, prerequisites
and scope. Separate narrative and code with ``# %%`` blocks. Add ``README.rst``
when creating a subsection.

Use fixed seeds, small synthetic inputs, and visible results. Include a useful
check such as a reconstruction residual or adjoint identity. Avoid downloads,
hidden datasets, blanket exception handling and installation commands in scripts.
Optional examples must fail clearly when explicitly selected without their
dependencies. Label training demonstrations and measured-data reproductions.

See the `Sphinx-Gallery authoring guide
<https://sphinx-gallery.github.io/stable/syntax.html>`_ and
`execution configuration
<https://sphinx-gallery.github.io/stable/configuration.html>`_. The site uses
`Sphinx Book Theme <https://sphinx-book-theme.readthedocs.io/>`_.

API generation
--------------

``docs/_ext/bartorch_api.py`` reads Python ASTs at build time, combines generated
tools with public handwritten overrides, and renders signatures and NumPy
docstrings with Sphinx's Python domain and Napoleon. It does not import bartorch,
run the wrapper generator or load native code. References describe the checkout,
not an unrelated installed wheel.

BART's ``help_str``, argument macros and option descriptions already feed
``build_tools/gen_tools.py``. That is the existing C-to-Python documentation
pipeline; arbitrary C comments do not necessarily describe a safe Python API.
Python overrides take precedence because they translate arguments and axes.
Documentation-only NumPy-style supplements live in ``docs/_docstrings/`` and
take precedence during rendering. PICS currently needs one because its initial
string-concatenation expression is not a Python docstring. Keep these supplements
aligned with the public signature; they do not change runtime ``help()``.
Supplement missing units, shapes, return contracts and exceptions in NumPy-style
wrapper docstrings during normal source development. Do not infer autograd
support or solver guarantees from C help text.
