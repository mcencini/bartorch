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

An encoding, and a solver driving it, is what most reconstructions here are.

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

Reach for `NoncartesianSense` off the grid and `CartesianSense` on it -- the
same operator over BART's FFT instead of a NUFFT, with `Sampling` chained on.
`WaveSense` is Wave-CAIPI, the same six operators `src/wave.c` chains in the
same order.  `Coils` is the sensitivity multiply on its own, for an encoding
whose transform is not a Fourier transform and so cannot be a SENSE operator.
Each is a single BART operator once built.

All four take several sets of maps -- ESPIRiT's second, ENLIVE's relaxed model
-- as `(sets, coils, *spatial)`, which is what `tools.ecalib` and
`tools.nlinv` return for `maps > 1`.  The image then carries the sets and the
samples do not: the encoding is $y_c = \sum_m S_{m,c} x_m$, contracted in
BART's own `md_ztenmul`.

`CartesianSense` and `WaveSense` read a temporal subspace when given a
`basis` -- T2 shuffling and Wave-Shuffling, the forward `pics -B` builds --
and with `toeplitz=True` the normal collapses the frames into one
coefficient-by-coefficient kernel, so an iteration never makes them.
`FieldCorrected` wraps any of these for off-resonance during the readout,
as a short sum of ordinary encodings rather than one chain.

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

The only functions on this page.  Everything else BART offers maps a tensor to
a tensor and is therefore an operator in its own right, which makes it a
class; these take operators and give back an operator.  Each is one BART
operator, so what a solver drives is a single operator rather than a list
walked per iteration.

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

Rearranging, reducing and restricting are operators like any other: each
composes with `@` and `+` and is solved by {mod}`bartorch.optim`.

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
