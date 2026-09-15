# Optimization

`bartorch.optim`.  A block is one step of a BART iteration as a torch module; a
solver loops a block to BART's schedule and is called as
`solver(y, A, x0=None)`; the function `optim.fista(y, A, term)` is that call in
one expression.

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
   maxeigen
```

A {class}`~bartorch.priors.ImplicitPrior` goes wherever a term goes.
{class}`NIHT` refuses: BART's own iteration asserts against the operator
`lsqr2` hands it, so no NIHT solve runs, `bart pics -R H` included.

## Blocks

`state = block.start(y, A, x0)` sets a run up, `state = block(state, A)` takes a
step, and `block.output(state, A)` is the image.  A solver is these three
calls in a loop, so a stack of frozen blocks answers with the solver's bits.
Step sizes and weights are parameters, frozen until `requires_grad_()`.

```python
blocks = nn.ModuleList(
    optim.FISTABlock(priors.ImplicitPrior(UNet(), sigma=0.05), step=0.9) for _ in range(10)
)
for block in blocks:
    block.step.requires_grad_()

state = blocks[0].start(kspace, A)
for block in blocks:
    state = block(state, A)
image = blocks[-1].output(state, A)
```

BART's proximal operators and the residual norms that steer the schedule
carry no derivative; everything else a step applies is recorded.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ISTBlock
   FISTABlock
   ADMMBlock
   PRIDUBlock
```

## Functional wrappers

`optim.fista(y, A, term, maxiter=30)` is `optim.FISTA(term, maxiter=30)(y, A)`.
Reach for the class to hand a solver to {class}`IRGNM` as `inner=`.

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
   irgnm
```

## Preconditioning

BART calls two unrelated things preconditioning.  `precond=` is
`lsqr2_create`'s `precond_op`, which every solver here takes; `pics --precond`
reformulates the data fidelity instead.

## Nonlinear least squares

BART has Gauss-Newton in two forms and {class}`IRGNM` is both: without
`inner=` it is `irgnm`, run inside the library, and with one it is `irgnm2`,
whose linearized problem goes to any solver here.

```python
optim.IRGNM(inner=optim.CG())                               # iter4_irgnm2, to the bit
optim.IRGNM(inner=optim.FISTA(priors.Wavelet(axes, 0.001)))   # moba -l1's shape
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
