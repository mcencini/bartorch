# Linear operators

`bartorch.linop`.  A linear operator maps between two C-order shapes.
{doc}`../explanation/encoding` states the model these operators implement.

Composition and combination produce a single BART operator rather than a Python
chain, so an iterative solver applies one operator per iteration without
returning to Python.  Applying an operator to a tensor that requires a gradient
records the application for autograd.

```{eval-rst}
.. currentmodule:: bartorch.linop
```

## MRI encoding operators

Each of these implements a complete SENSE forward model -- coil sensitivities,
transform and sampling -- for one sampling regime: {func}`CartesianSense` for
Cartesian sampling, {class}`NoncartesianSense` for non-Cartesian trajectories,
and {func}`WaveSense` for Wave-CAIPI.  {func}`FieldCorrected` wraps any of them
with off-resonance correction by time segmentation.  Other encoding models are
built by composing these with the operators below.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   CartesianSense
   NoncartesianSense
   WaveSense
   FieldCorrected
```

## Linear operators and operator algebra

`A @ B` composes, `A + B` and `A - B` add, `c * A` scales, `A ** n` repeats,
`A.H` is the adjoint and `A.T` the transpose, `A.gram()` and `A.cogram()` are
the normal operators `A^H A` and `A A^H`, and `A[key]` restricts the codomain
as indexing a tensor does.  The concrete types these return are implementation
detail; each is a {class}`LinearOperator`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearOperator
```

## Elementary operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Identity
   Zero
   Diagonal
   ComponentDiagonal
   Conj
   Real
   FFT
   NUFFT
   MultiplySum
```

## Matrix, convolution and finite-difference operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Matrix
   Convolve
   Gradient
```

## Stacking and block composition

Each of these builds a single BART operator over its operands rather than a
Python container traversed once per iteration.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   concatenate
   stack
   hstack
   block_diag
   block
```

## Shape and indexing operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Reshape
   Transpose
   Permute
   Flip
   Roll
   Pad
   Resize
   Extract
   Hankel
   Sum
   ScaledSum
   Mean
   Repeat
```

## User-defined operators

{meth}`LinearOperator.from_callbacks` builds an operator from Python functions
for the forward, the adjoint and, where a cheaper form is available, the
normal.  BART reaches such an operator through callbacks, at the cost of one
crossing into Python per application.
