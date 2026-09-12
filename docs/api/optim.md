# Optimization

`bartorch.optim`.  A solver is configured once and called as
`solver(y, A, x0=None)`; the iteration runs inside BART, as the one `pics` or
`nlinv` runs.

```{eval-rst}
.. currentmodule:: bartorch.optim
```

## Linear least squares

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   CG
   Tikhonov
```

`CG` minimizes $\|Ax-y\|^2 + \lambda\|x\|^2 + \sum_i w_i\|G_ix-b_i\|^2$.
The first two are BART's own: `iter2_conjgrad` takes one weight and asserts it
is handed no regularizing operators and no biases, and `lsqr2_create` builds
$A^HA+\lambda I$.  The rest are `Tikhonov` terms, which are not passed to the
iteration at all -- they are stacked under the encoding, so that what BART
solves is still an ordinary least-squares problem.

Stacking builds the normal operator rather than letting BART derive it, as
$A^HA+\sum_i w_iG_i^HG_i$ over each operator's own normal.  A Toeplitz
encoding therefore stays one inside a regularized solve: `A.gram()` is the
point-spread convolution, not the transform and its adjoint.

## Regularized least squares

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   IST
   FISTA
   ADMM
   PRIDU
   NIHT
   EulerMaruyama
```

## Nonlinear least squares

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   IRGNM
```

## Data scaling

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   data_scaling
```
