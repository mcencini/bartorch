# Linear operators

`bartorch.linop`.  An operator maps between two C-order shapes; operators
compose with `@` and `+` into a single BART operator, and applying one to a
tensor that requires a gradient records it for autograd.

```{eval-rst}
.. currentmodule:: bartorch.linop
```

## Linear operator class

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearOperator
```

## Composition

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Compose
   Add
   Adjoint
```

## Elementary operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FFT
   Diagonal
   Sampling
   MultiplySum
```

## MRI encoding

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NUFFT
   Sense
```

## Python-defined operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Callback
```
