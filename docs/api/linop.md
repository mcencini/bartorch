# Linear operators

`bartorch.linop`.  An operator maps between two C-order shapes.  They compose
and combine into a single BART operator through the algebra below, so what a
solver drives is one operator in BART's own loop.  Applying one to a tensor
that requires a gradient records it for autograd.

```{eval-rst}
.. currentmodule:: bartorch.linop
```

## MRI encoding

An encoding and a solver driving it is what most reconstructions here are.
Reach for `NoncartesianSense` off the grid, `CartesianSense` on it,
`WaveSense` for Wave-CAIPI, and `Coils` for the sensitivity multiply alone.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   CartesianSense
   NoncartesianSense
   WaveSense
   Coils
   FieldCorrected
   Sampling
```

## Linear operator class

`A @ B` composes, `A + B` and `A - B` add, `c * A` scales, `A ** n` repeats,
`A.H` and `A.T` transpose, `A.gram()` and `A.cogram()` are the normal
operators, and `A[key]` restricts the output as indexing a tensor does.  What
each returns is private.

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

## Filtering and differencing

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Matrix
   Convolve
   Gradient
```

## Combining operators

The only functions on this page: everything else maps a tensor to a tensor and
is therefore an operator, which makes it a class.  Each of these is one BART
operator, not a list walked per iteration.

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

## Shape

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

## Python-defined operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Callback
```
