# Linear operators

`bartorch.linop`.  An operator maps between two C-order shapes.  They compose
and combine into a single BART operator through the algebra below -- every one
of those is a BART constructor applied to BART operators, so what a solver
drives is one operator in BART's own loop, with no arithmetic in Python
between the steps.  Applying an operator to a tensor that requires a gradient
records it for autograd.

```{eval-rst}
.. currentmodule:: bartorch.linop
```

## MRI encoding

These are what most reconstructions here are: an encoding, and a solver driving
it.  They come first because they are what the rest of the page is for.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   CartesianSense
   NoncartesianSense
   WaveSense
   FieldCorrected
   Sampling
```

`NoncartesianSense` is BART's own operator -- the coil batching and the
Toeplitz normal are what make it what it is.  `CartesianSense` is that same
operator over BART's FFT rather than a NUFFT, with `Sampling` chained on;
`WaveSense` is a composition of operators BART already has, chained by
`linop_chain`, and is the same six in the same order that `src/wave.c` chains.
Each is a single BART operator once built.

`FieldCorrected` wraps any of them and is a sum of chains rather than one:
off-resonance during the readout is a different transform per sample, and time
segmentation stands in for it with a short sum of ordinary encodings.  The
coefficients come from `mri-nufft`; everything applied is BART's.

## Linear operator class

An operator is combined with others through the algebra its class defines,
rather than by naming a combining class: `A @ B` composes, `A + B` and
`A - B` add, `c * A` scales, `A ** n` repeats, `A.H` and `A.T` transpose,
`A.gram()` and `A.cogram()` are the normal operators, and `A[key]` restricts
its output the way indexing a tensor does.  What each returns is private,
because the algebra promises the operator, not the shape of the tree it is
made of.

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

The only functions here.  Everything else BART offers maps a tensor to a
tensor and is an operator in its own right, which makes it a class; these take
operators and give back an operator, which is the one thing that cannot be
written as arithmetic or as indexing.  Each is one BART operator, so what a
solver drives is a single operator rather than a list walked per iteration.

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

Rearranging, reducing and restricting are operators like any other: each maps
a tensor to a tensor, so each is a class, composes with `@` and `+`, and is
solved by {mod}`bartorch.optim`.  Only something that takes operators and
gives back an operator would be a function here.

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
