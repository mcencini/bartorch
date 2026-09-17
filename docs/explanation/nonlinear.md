# When the operator is not linear

{doc}`inverse-problems` assumed the forward operator was known and linear. Two
common reconstructions violate that assumption in the same way, and are solved
by the same method.

## Two problems with the same shape

**The sensitivities are unknown.** ESPIRiT estimates them from a fully sampled
neighbourhood of the k-space centre, and an acquisition that provides no such
neighbourhood provides no way to estimate them separately. Treating them as
unknowns gives

$$
y_c = P F (S_c \cdot x) ,
$$

which is bilinear: linear in $x$ for fixed $S$, linear in $S$ for fixed $x$,
and not linear in the pair. This is **nonlinear inversion**, BART's `nlinv`.

**The image is a function of physical parameters.** A quantitative experiment
measures a series of contrasts whose dependence on the tissue parameters is
known — a mono-exponential decay in $T_2$, an inversion recovery in $T_1$, a
Bloch simulation of an arbitrary sequence. Writing $M$ for that signal model
and $\theta$ for the parameter maps,

$$
y_{c,e} = P_e F (S_c \cdot M_e(\theta)) ,
$$

and the unknown is $\theta$. This is **model-based reconstruction**, BART's
`moba`.

In both, the operator $F$ mapping the unknowns to the data is nonlinear, and
what a solver needs from it is not a matrix but three things: its value
$F(x)$, its derivative at a point as a linear operator $DF_x$, and the adjoint
of that derivative. A {class}`~bartorch.nlop.NonlinearOperator` is exactly that
triple.

## Gauss-Newton with decreasing regularization

Both problems are solved by **iteratively regularized Gauss-Newton** (IRGNM).
Each step replaces the operator by its linearization at the current iterate and
solves the resulting linear problem:

$$
u^k = \arg\min_u \; \| DF_{x^k} u - r^k \|_2^2 + \alpha_k \| u \|_2^2 ,
\qquad x^{k+1} = x^k + u^k ,
$$

with $r^k = y - F(x^k)$ the residual. Each inner problem is a regularized
linear least-squares problem of the kind {doc}`inverse-problems` describes, and
is solved the same way.

The distinctive part is $\alpha_k$, which starts large and is divided by a
constant — two, by default — after every step. The early steps are heavily
regularized and move the iterate in the well-determined directions only; the
later ones let the poorly-determined directions in. This decreasing
regularization carries the method through a problem whose linearization is
ill-conditioned everywhere, and it means the iteration count is a
regularization parameter rather than a convergence threshold: stopping early
leaves a smoother answer, running longer eventually lets noise in.

{class}`~bartorch.nlop.IRGNM` is that loop, and
{class}`~bartorch.nlop.IRGNMBlock` is one step of it as a torch module. The
inner problem can go to the conjugate gradients inside BART, as in `nlinv`, or
to a solver from {mod}`bartorch.optim`, whose regularization terms then apply
to the step.

## Identifiability, and what makes the factorization unique

The bilinear problem has a symmetry: $S_c \cdot x$ is unchanged by
$S_c \mapsto \gamma S_c$, $x \mapsto x/\gamma$ for any nonzero function
$\gamma$. Nothing in the data distinguishes the factors, so a solver would be
free to put all of the anatomy in the sensitivities and none in the image.

What breaks the symmetry is prior knowledge: the sensitivities are smooth, the
image is not. `nlinv` imposes this inside the model rather than beside it. The
coil unknown is not the sensitivity map but a k-space representation
$\hat{s}$ of it, and the map follows by a **Sobolev weighting**

$$
S = \mathcal{F}^{-1}\!\left[ (1 + a |k|^2)^{-b/2} \, \hat{s} \right] ,
$$

which attenuates the high spatial frequencies of whatever the solver produces.
A step in the unknown is therefore a smooth change in the map by construction,
and the joint problem needs no separate penalty on the coils. What remains
undetermined is the overall scale, which is why a nonlinear inversion is
reported after normalizing by the root sum of squares of the estimated maps.

The parameter problem has no such symmetry — the signal model fixes what each
map means — but it is nonconvex, so the starting point matters and a fit can
settle in a local minimum. Bounds on the parameters keep the iterates
physical, and the models in {mod}`bartorch.nlop` carry them in a transformed
parameterisation rather than as constraints, which is why the maps a fit
returns are read back with
{meth}`~bartorch.nlop.SignalModel.split` rather than taken directly from the
state.

## What model-based reconstruction buys

The alternative to putting the model in the operator is the two-step route:
reconstruct the series of contrasts, then fit the model voxel by voxel. The
two differ in what the reconstruction is allowed to use.

In the two-step route each contrast is reconstructed on its own, from its own
undersampled data, and nothing in that reconstruction uses the relation between
the contrasts. The fit that follows is given whatever artefact the
reconstruction left, with no way to distinguish it from signal.

In the model-based route the unknowns are the parameter maps — three maps
rather than eight images, for a multi-echo experiment — and every echo
constrains all of them. The problem is better determined for the same data, and
the regularization applies to the maps rather than to the images, which is
usually where the prior knowledge is.
{doc}`../auto_examples/04-model-based/02-quantitative-models` runs both routes
on the same data with the same model and the same solver.

A subspace reconstruction sits between the two: it is linear, so it uses the
solvers of {doc}`inverse-problems`, and it constrains the series through a
basis estimated from a simulated dictionary rather than through the model
itself. It gives up the model's exact parameterisation and keeps a convex
problem. {doc}`../auto_examples/03-applications/02-subspace-t1-mapping` is that
route on an inversion-recovery experiment.

## Signal models and differentiation

The signal models in {mod}`bartorch.nlop` —
{func}`~bartorch.nlop.InversionRecovery`, {func}`~bartorch.nlop.MultiEcho`,
{func}`~bartorch.nlop.Bloch` — are
[TorchSim](https://github.com/FiRMLAB-Pisa/torchsim) simulators presented as
BART nonlinear operators. A signal model is voxel-diagonal: the signal of a
voxel depends on that voxel's parameters alone, so one forward-mode pass gives
the derivative of the whole volume whatever the number of parameters, and no
Jacobian is ever formed. {class}`~bartorch.nlop.TorchOperator` does the same
for any differentiable PyTorch function, taking its derivative and adjoint from
autograd, which is how a model the library does not ship is fitted by the same
solver.

Nonlinear operators compose with `@`, and either side may be a linear operator,
which is how a signal model is placed in front of an encoding. The derivative
of the composition at a point is the encoding applied to the derivative of the
model, and the Gauss-Newton step needs nothing else.

Differentiation runs the other way too. A Gauss-Newton step is itself
differentiable — by the data, by the iterate, by the regularization centre and
by $\alpha$ — so an unrolled network trains through it, and the same holds of
the proximal steps in {mod}`bartorch.optim`. What is not differentiable
is a BART proximal operator, which has no implemented backward pass and raises
rather than contributing a wrong gradient;
{class}`~bartorch.priors.ImplicitPrior` substitutes a differentiable denoiser
where one is needed.

## References

Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
nonlinear inversion -- joint estimation of coil sensitivities and image
content. *Magn Reson Med* 60(3):674-682 (2008).

Bakushinsky AB, Kokurin MY. *Iterative methods for approximate solution of
inverse problems.* Springer (2004).

Block KT, Uecker M, Frahm J. Model-based iterative reconstruction for radial
fast spin-echo MRI. *IEEE Trans Med Imaging* 28(11):1759-1769 (2009).

Wang X, Tan Z, Scholand N, Roeloffs V, Uecker M. Physics-based reconstruction
methods for magnetic resonance imaging. *Phil Trans R Soc A* 379:20200196
(2021).
