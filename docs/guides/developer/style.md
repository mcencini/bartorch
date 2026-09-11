# Style

Python follows the repository's ruff configuration: 100-column lines,
formatter-managed layout and sorted imports.  Use descriptive names and public
type hints.  Keep tensor shapes, devices, normalization, mutation and ownership
explicit at native boundaries.  Use plain C in the exported ABI, follow the
existing local conventions, and keep upstream BART unchanged.

A wrapper around a BART command calls BART; it does not reimplement BART's
computation in torch.  Torch is for reshaping and marshalling around the call.

Docstrings follow {doc}`documentation`.  Pages are Markdown (MyST); gallery
examples are Python scripts with narrative blocks.  Document implemented
behaviour, separate proposed methods from available APIs and paper
reproductions from teaching phantoms, and label plots with units and
normalization.
