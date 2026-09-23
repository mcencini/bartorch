# Coding style

## Python

Python follows the repository's ruff configuration: 100-column lines,
formatter-managed layout, sorted imports and the `E`, `F`, `W` and `I` rule
sets, applied to `src/`, `tests/` and the gallery examples.  Public functions
carry type hints.  Shapes, devices, normalization, mutation and ownership are
explicit at the native boundary.

A wrapper around a BART command calls BART; it does not reimplement BART's
computation in PyTorch, which is used for reshaping and marshalling around the
call.  The exceptions are listed in `AGENTS.md`: the substitutions that are
faster than BART, and signal simulation, which is TorchSim's.

## C

The exported ABI in `src/csrc/include/bartorch.h` is plain C: no complex types
and no variable-length arrays.  Local conventions of each file are followed,
and BART's own sources are not modified.

## Docstrings

Docstrings are NumPy style and follow {doc}`documentation`.  Every documented
parameter with a default states it in its type line, as
`maxiter : int, default=30`, matching the signature; `tests/test_docstrings.py`
checks this for every public object.  Write for a reader of the code as it is:
no descriptions of what the code used to do.
