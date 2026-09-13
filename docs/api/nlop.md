# Nonlinear operators

`bartorch.nlop`.  A nonlinear operator carries its derivative and the adjoint
of that derivative at the last evaluated point, which is what BART's
Gauss-Newton solver and torch's autograd both use.

An operator maps *many* inputs to *many* outputs, as BART's `nlop_s` does: the
model `nlinv` inverts takes an image and a set of coil profiles and returns
k-space.  {attr}`~bartorch.nlop.NonlinearOperator.ishapes` and
{attr}`~bartorch.nlop.NonlinearOperator.oshapes` give the shape of each
argument, and `ishape` and `oshape` are the whole of it when there is one of
each.  Arguments are counted BART's way throughout -- outputs first, then
inputs.

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

The algebra of `nlops/chain.h`.  {func}`combine` puts two operators side by
side and is where every other combination starts; {func}`chain` feeds one
output into one input; {meth}`~NonlinearOperator.link` ties an output of one
operator back into an input of another, and
{meth}`~NonlinearOperator.dup` makes two inputs one.

BART applies a combination back to front -- in `combine(a, b)` it is `b` that
runs first -- so an operator that produces a value goes *second*, and the one
that consumes it first.  {func}`chain` arranges that itself; a
{meth}`~NonlinearOperator.link` the other way round is refused rather than
left to read a buffer nothing has written.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   chain
   combine
   Chain
   FromLinear
   Derivative
```

## Basic operators

The tensor product and the elementwise maps, each one BART constructor.
{class}`Multiply` is the two-input product every model is built out of.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Multiply
   Divide
   Weighted
   Constant
   Exp
   Log
   Sqrt
   Power
   Add
   Inverse
   Abs
   SmoothAbs
   Phase
   Sum
   RootSumOfSquares
```

## MRI encodings

The nonlinear variants of the linear encodings: the coils are a second
unknown rather than a fixed tensor, which is what `nlinv` inverts.
{class}`NonlinearSense` is BART's own `noir` model -- the same code `nlinv`
runs, so a fit driven from here is the same arithmetic -- and
{func}`CoilSense` is the general recipe over any linear encoding.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearSense
   CartesianSense
   NoncartesianSense
   CoilSense
```

## Signal models

A model-based reconstruction is a signal model under an encoding, and the
model is the part that changes with the sequence.  {class}`FromTorchSim`
turns any [TorchSim](https://github.com/FiRMLAB-Pisa/torchsim) simulator into
a BART operator -- its value, its Jacobian-vector product and its adjoint
product, which is exactly the three things `nlop_s` asks for, and none of
which ever builds a Jacobian.

{func}`InversionRecovery`, {func}`MultiEcho` and {func}`Bloch` are `moba`'s
families written on TorchSim's own simulators.  They are not BART's models:
`moba` writes each one out in C with its own parameterisation.  Where the two
agree on the physics they agree on the numbers -- the multi-echo decay matches
`bart signal -S` to single precision -- and where `moba` reparameterises they
do not.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FromTorchSim
   InversionRecovery
   MultiEcho
   Bloch
```

## A Gauss-Newton step, which BART already differentiates

`noir/model_net.c` builds one iteration of `nlinv` as an `nlop`,

$$x_{n+1} = x_n + (DF^H DF + \alpha)^{-1}\left[DF^H (y - F(x_n)) - \alpha (x_n - x_0)\right]$$

and it builds it out of `nlop`s throughout: the forward model, the derivative
*as a function of the linearisation point*, the adjoint, and `norm_inv`'s
implicitly differentiated inverse of the normal operator.  So the step has a
derivative of its own -- by the data, the iterate, the regularisation centre
and the weight, second-order terms included -- and {class}`GaussNewton` is
that operator rather than a reimplementation of it.  BART reconstructs with
it in `networks/nlinvnet.c`; a denoiser between two of these is NLINV-Net.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GaussNewton
```

Three things are worth knowing before using it.

**The model has no sampling pattern until it is given one, and it is given one
as a side effect of the gridding.** `noir_adjoint_fft_fun` calls
`linop_gdiag_set_diag(model->lop_pattern, ...)` on its way past, and off the
grid `noir_adjoint_nufft_fun` calls `nufft_update_traj`. So `prepare()` has to
be applied to *this* operator before a step is, and two operators do not share
a model. A step applied to a model that never got a pattern reads a diagonal
nothing has written, which is a segmentation fault rather than an error, so
the operator refuses instead.

It follows that the pattern is state. An operator carries whichever pattern
its last `prepare()` set, so in a training loop `prepare()` belongs in the
forward pass beside the step, not once at the start -- and the operator must
not be prepared elsewhere between a forward pass and its backward pass, for
the same reason a nonlinear operator must not be evaluated there.

**The batch is BART's own.** Everywhere else here a leading axis is applied
item by item from Python; this is the one operator that stacks a batch inside
the library, because `nlinvnet` needed it. `batch=` says how many independent
copies of the model to build, and that count is the leading axis of every
argument.

**The shapes are BART's sixteen axes, written short.** The operator records
what BART reports, because that is what the arity check holds it to; what a
caller passes and what comes back is `shapes` and `output_shapes`, which are
the same tuples without the run of empty axes between the batch and the image.
A run of singletons changes no strides, so moving between the two is a reshape
and not a copy.

The iterate is the image and the coil coefficients laid end to end; `start()`
makes the one BART starts from, `split()` and `join()` take it apart and put it
back, and `decompose()` takes it apart *through* the model's transforms, so
what comes back is coil profiles rather than the coefficients that were fitted.

## Python-defined operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Callback
   FromTorch
```
