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

## Linear operator class

An operator is combined with others through the algebra its class defines,
rather than by naming a combining class: `A @ B` composes, `A + B` and
`A - B` add, `c * A` scales, `A ** n` repeats, `A.H` and `A.T` transpose,
`A.gram()` and `A.cogram()` are the normal operators.  What each returns is
private, because the algebra promises the operator, not the shape of the tree
it is made of.

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
   Conj
   Real
   FFT
   NUFFT
   MultiplySum
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
   Sum
   ScaledSum
   Mean
   Repeat
```

## MRI encoding

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Sampling
   Sense
```

## Python-defined operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Callback
```
