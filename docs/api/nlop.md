# Nonlinear operators

`bartorch.nlop`.  A nonlinear operator carries its derivative and the adjoint
of that derivative at the last evaluated point, which is what BART's
Gauss-Newton solver and torch's autograd both use.

```{eval-rst}
.. currentmodule:: bartorch.nlop
```

## Nonlinear operator class

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearOperator
```

## Composition

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Chain
   FromLinear
```

## Python-defined operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Callback
   FromTorch
```
