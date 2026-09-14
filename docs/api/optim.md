# Optimization

`bartorch.optim`.  A solver is configured once and called as
`solver(y, A, x0=None)`; the iteration is BART's, the one `pics` or `nlinv`
runs.  `solver.in_library(y, A)` runs BART's own loop instead, and answers
with the same bits.

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
   maxeigen
```

## Functional wrappers

`optim.fista(y, A, term, maxiter=30)` is `optim.FISTA(term, maxiter=30)(y, A)`,
and is the ordinary way to run one.  Reach for the class when the solver has
to be *held*: to unroll it, to drive it to a fixed point, or to hand it to
{class}`IRGNM` as `inner=`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cg
   ist
   fista
   admm
   pridu
   niht
   eulermaruyama
   irgnm
```

A `deepinv` prior or a bare denoiser goes wherever a {mod}`bartorch.prox` term
goes, which the classes do not take.  `niht` and {class}`NIHT` refuse: BART's
own iteration asserts against the operator `lsqr2` hands it, so no NIHT solve
runs, `bart pics -R H` included.

## Unrolling

{meth}`~CG.unrolled` makes a solver a torch network and {meth}`~CG.fixed_point`
drives it to a fixed point.  Both build the step themselves; there is no
iteration class to name.

```python
net = optim.FISTA(denoiser, maxiter=10, step=0.9).unrolled(
    (1, 256, 256), trainable=["stepsize"]
)
image = net(kspace[None], bartorch.to_deepinv(A))
```

Every operator a step applies is recorded, so the graph is over BART's own
arithmetic.  Two things are deliberately not in it: BART's proximal operators,
which carry no derivative, and the residual norms that steer $\rho$, $\tau$ and
the stopping test.

## Preconditioning

BART calls three unrelated things preconditioning.  `precond=` is
`lsqr2_create`'s `precond_op`, which every solver here takes; `pics --precond`
reformulates the data fidelity instead; and {class}`EulerMaruyama`'s
`sampler_precond=` is the one preconditioned solve BART really runs.

## Nonlinear least squares

BART has Gauss-Newton in two forms and {class}`IRGNM` is both: without
`inner=` it is `irgnm`, run inside the library, and with one it is `irgnm2`,
whose linearized problem goes to any solver here.

```python
optim.IRGNM(inner=optim.CG())                               # iter4_irgnm2, to the bit
optim.IRGNM(inner=optim.FISTA(prox.Wavelet(axes, 0.001)))   # moba -l1's shape
```

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
