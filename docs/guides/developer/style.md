# Style

Python follows the repository's ruff configuration: 100-column lines,
formatter-managed layout and sorted imports.  Use descriptive names and public
type hints.  Keep tensor shapes, devices, normalization, mutation and ownership
explicit at native boundaries.  Use plain C in the exported ABI, follow the
existing local conventions, and keep upstream BART unchanged.

A wrapper around a BART command calls BART; it does not reimplement BART's
computation in torch.  Torch is for reshaping and marshalling around the call.

Documentation and docstrings follow {doc}`documentation`, which is the
editorial policy for this project.  Pages are Markdown (MyST).  Document
implemented behaviour, and separate proposed methods from available APIs.
