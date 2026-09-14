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

Two arguments of the same shape can still be held at different *ranks*.
`nlop_chain2` and `nlop_link` compare `iovec`s, and an `iovec` carries its
rank: BART builds each of its own operators at whatever rank it needs -- the
state of {class}`GaussNewton`'s step is two axes long -- while an operator
defined here through {class}`Callback` or {class}`FromTorch` is built at
DIMS. Padding a shape with ones is not a change to it, so {func}`chain` and
{meth}`~NonlinearOperator.link` write the shorter side out to match rather
than refusing with BART's `Cannot chain args 0 -> 0!`.
{meth}`~NonlinearOperator.reshape_input` and
{meth}`~NonlinearOperator.reshape_output` are the same thing said by hand,
for the cases where the two shapes differ by more than padding.

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

The fits agree as well as the curves.  Measurements `bart signal` produced,
handed to `bart mobafit`'s pixel-wise Gauss-Newton and to
{class}`~bartorch.optim.IRGNM` over the model here, come back with the same
relaxation time, and under noise the two land together rather than each near
the truth.  What differs is the variables: `mobafit -I` reports
`(M0, R1, c)` where this reports a T1 and an amplitude, and `mobafit -m3`
carries an `fB0` that TorchSim's multi-echo model has nowhere to put -- its
`offset` is an additive baseline, not a frequency.

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

Three things about it are easy to trip over, and each has a longer account in
the class's own documentation.

**The pattern is state, and `prepare()` is what sets it.** BART writes the
sampling pattern into the model as a side effect of the gridding, so
`prepare()` has to be applied to *this* operator before a step is, two
operators do not share a model, and in a training loop `prepare()` belongs in
the forward pass beside the step rather than once at the start.

**The batch is BART's own.** Everywhere else here a leading axis is applied
item by item from Python; this is the one operator that stacks a batch inside
the library.  `batch=` is the leading axis of every argument.

**The shapes are BART's sixteen axes, written short.** `shapes` and
`output_shapes` are what a caller passes and gets back; `ishapes` and
`oshapes` stay what BART reports, because that is what the arity check holds
the operator to.

The iterate is the image and the coil coefficients laid end to end; `start()`
makes the one BART starts from, `split()` and `join()` take it apart and put it
back, and `decompose()` takes it apart *through* the model's transforms, so
what comes back is coil profiles rather than the coefficients that were fitted.

An unrolled network can be composed into a *single* `nlop`: chain the cells,
with whatever stands between them, and BART drives the whole thing and crosses
into Python once a step for the prior alone.

```python
prior = nlop.FromTorch(denoise, first.state_shape, first.state_shape)
whole = nlop.chain(nlop.chain(first, prior, output=0, input=0), second, output=0, input=1)
```

That operator differentiates by its own arguments -- data, iterate, centre,
weight -- with `FromTorch` answering for the prior through `torch.func`'s jvp
and vjp.

**The prior's weights train through it too, if they are arguments.** A weight
the denoiser *closes over* is not an argument of anything BART knows about, so
no gradient reaches it. Give the function the weights instead and they become
inputs of the network:

```python
prior = nlop.FromTorch(lambda x, w: w * x, [first.state_shape, ()], first.state_shape)
whole = nlop.chain(nlop.chain(first, prior, output=0, input=0), second, output=0, input=1)
...
whole(y, x0, alpha, weight, ...).abs().square().sum().backward()   # reaches `weight`
```

BART applies the whole network and torch reaches every one of its arguments,
the prior's parameters included. A real parameter rides in the real part of a
complex one, because BART's operators are complex throughout, so its gradient
comes back complex and the real part is the one to take.

A real denoiser keeps its parameters in several tensors of several shapes, and
{class}`Parameters` packs them into the one complex vector BART can carry and
unpacks them again.  The packed vector is the thing to hold as a
`torch.nn.Parameter`: it is what the operator differentiates.  How the
denoiser sees the iterate is the caller's to say -- usually the image half of
the state, shaped as an image.

A gradient in the *coil* half of the state is unreliable at BART's default
coil weighting, which is a sixteenth power; {class}`GaussNewton`'s
documentation says why and what to use instead.

## Python-defined operators

One argument or many.  A function of several tensors becomes an `nlop` of
several inputs -- `FromTorch(fn, [shape, ()], shape)` for a function of a
tensor and a scalar -- which is what lets a denoiser's weights be *arguments*
of a BART graph rather than something the function closed over, and so what
lets a gradient reach them when the graph is BART's to apply.

With several arguments the derivative and its adjoint are asked for a pair:
`derivative(o, i, dx)` is the derivative of output `o` by input `i`, and
`adjoint(o, i, dy)` the adjoint of that. {class}`FromTorch` works them out
with `torch.func`, zeroing the tangent in every argument but the one it was
asked about. The single-argument forms are unchanged and still go through
BART's single-argument constructor.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Callback
   FromTorch
   Parameters
```
