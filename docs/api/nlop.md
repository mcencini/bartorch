# Nonlinear operators

`bartorch.nlop` represents a nonlinear operator $F$ from one or more inputs to
one or more outputs, with its derivative $DF_x$ at a point and the adjoint of
that derivative.  `F(*x)` evaluates it, `F.linearize(*x)` returns $DF_x$ as a
{obj}`~bartorch.linop.LinearOperator`, `a @ b` composes (applying `b` first,
either side possibly linear), and `F.partial(i, value)` fixes input `i`.
{doc}`../explanation/nonlinear` describes the reconstruction problems these
operators model, and {doc}`../explanation/differentiation` how they and the
Gauss-Newton steps enter autograd.

```{eval-rst}
.. currentmodule:: bartorch.nlop
```

## Operator class

| Object | Description |
| --- | --- |
| {obj}`~bartorch.nlop.NonlinearOperator` | Base class: evaluation, derivative and its adjoint, composition, partial application, linearization |
| {obj}`~bartorch.nlop.TorchOperator` | Nonlinear operator from a differentiable PyTorch function, with derivative and adjoint from autograd |

## MRI encoding models

| Object | Inputs | Model |
| --- | --- | --- |
| {obj}`~bartorch.nlop.NonlinearSense` | Image, Sobolev-weighted coil coefficients | BART's `noir` model, $y_c = PF(S_c x)$ with sensitivities $S_c$ estimated jointly |
| {obj}`~bartorch.nlop.CartesianSense` | Image, coil coefficients | {obj}`~bartorch.nlop.NonlinearSense` on a Cartesian grid |
| {obj}`~bartorch.nlop.NoncartesianSense` | Image, coil coefficients | {obj}`~bartorch.nlop.NonlinearSense` along a trajectory |
| {obj}`~bartorch.nlop.CoilSense` | Image, sensitivities | Product of image and unweighted sensitivities in front of any linear encoding |

## Signal models

Quantitative signal models evaluated by
[TorchSim](https://github.com/FiRMLAB-Pisa/torchsim); each maps parameter maps
to one image per contrast.

| Object | Unknowns | Signal |
| --- | --- | --- |
| {obj}`~bartorch.nlop.SignalModel` | Any TorchSim model's parameters | Base class: a TorchSim `ModelOperator` as a nonlinear operator |
| {obj}`~bartorch.nlop.InversionRecovery` | $T_1$, optional complex amplitude | Inversion recovery at a series of inversion times |
| {obj}`~bartorch.nlop.MultiEcho` | $T_2$ or $T_2^*$, optional complex amplitude | Mono-exponential decay at a series of echo times |
| {obj}`~bartorch.nlop.Bloch` | Any simulated tissue property | Bloch simulation of a TorchSim sequence |

## Elementary operators

| Object | Description |
| --- | --- |
| {obj}`~bartorch.nlop.Multiply` | Pointwise product of two inputs, with broadcasting |
| {obj}`~bartorch.nlop.Divide` | Pointwise quotient of two inputs |
| {obj}`~bartorch.nlop.Weighted` | $a x + b z$ of two inputs |
| {obj}`~bartorch.nlop.Constant` | Operator of no inputs returning a fixed tensor |
| {obj}`~bartorch.nlop.Exp` | $e^x$ |
| {obj}`~bartorch.nlop.Log` | $\log x$ |
| {obj}`~bartorch.nlop.Sqrt` | $\sqrt{x}$ |
| {obj}`~bartorch.nlop.Power` | $x^p$ |
| {obj}`~bartorch.nlop.Add` | $x + c$ |
| {obj}`~bartorch.nlop.Inverse` | $1/x$ |
| {obj}`~bartorch.nlop.Abs` | $\lvert x \rvert$ |
| {obj}`~bartorch.nlop.SmoothAbs` | $\sqrt{\lvert x \rvert^2 + \epsilon}$ |
| {obj}`~bartorch.nlop.Phase` | $x / \lvert x \rvert$ |
| {obj}`~bartorch.nlop.SumOfSquares` | $\sum \lvert x \rvert^2$ over axes |
| {obj}`~bartorch.nlop.RootSumOfSquares` | $\sqrt{\sum \lvert x \rvert^2}$ over axes |

## Gauss-Newton methods

| Object | Description |
| --- | --- |
| {obj}`~bartorch.nlop.IRGNM` | Iteratively regularized Gauss-Newton solver for $F(x) = y$ |
| {obj}`~bartorch.nlop.IRGNMBlock` | One Gauss-Newton step as a `torch.nn.Module`, for unrolling |
| {obj}`~bartorch.nlop.irgnm` | Functional form of {obj}`~bartorch.nlop.IRGNM` |

`IRGNM(inner=None)` solves each linearized problem by conjugate gradients
inside BART, as `nlinv` does; `IRGNM(inner=solver)` passes it to a solver from
{mod}`bartorch.optim`, whose regularization terms then apply to the step.

The examples {doc}`../auto_examples/04-model-based/01-nonlinear-inversion` and
{doc}`../auto_examples/04-model-based/02-quantitative-models` use these objects
in complete reconstructions.
