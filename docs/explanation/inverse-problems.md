# Inverse problems and their solvers

An MRI reconstruction estimates an image from measurements that do not
determine it. This page states that estimation problem, the conditions under
which it has a useful answer, and the algorithms that compute one.

## The measurement model

Write $x$ for the unknown image, $y$ for the measured data and $A$ for the
operator that says what data a given image would have produced. The
measurement is

$$
y = A x + \varepsilon ,
$$

with $\varepsilon$ the noise. $A$ is the **forward operator**, or in MRI the
**encoding operator**; what it consists of is the subject of
{doc}`encoding`. Recovering $x$ from $y$ is the **inverse problem**.

Three properties of $A$ decide what can be done with it.

**It is linear**, for a fixed set of coil sensitivities: the signal of a sum of
objects is the sum of their signals. A linear operator is determined by its
action, and never has to be formed as a matrix — for a $256^2$ image with
eight channels the matrix would have of order $10^{10}$ entries, while applying
the operator costs a few Fourier transforms.

**It has an adjoint.** The adjoint $A^H$ is defined by
$\langle Ax, y\rangle = \langle x, A^H y\rangle$ for all $x$ and $y$; for a
matrix it is the conjugate transpose. Every algorithm below is written in terms
of $A$ and $A^H$ alone, so one solver serves every encoding.
Applying the adjoint to the data, $A^H y$, is the cheapest thing that can be
called a reconstruction, and for Cartesian sampling it is the zero-filled
inverse Fourier transform followed by the coil combination.

**It is ill-conditioned, and may not be invertible at all.** An accelerated
acquisition measures fewer k-space positions than the image has voxels. Several
receive channels measure each position, so the number of equations can still
exceed the number of unknowns; whether the system determines the image depends
on how differently the channels see the voxels that alias onto each other. Where
the count does fall short, $A$ has a null space — there are images
$x_0 \neq 0$ with $A x_0 = 0$, and $x$ and $x + x_0$ explain the data equally
well — and where it does not, $A$ is merely ill-conditioned, and inverting it
amplifies the noise by the inverse of its smallest singular value.

## Least squares and its failure

The estimator that ignores this is least squares,

$$
\hat{x} = \arg\min_x \; \| A x - y \|_2^2 ,
$$

whose solutions satisfy the **normal equations** $A^H A \hat{x} = A^H y$. The
operator $A^H A$ is the **normal operator**, or Gram operator; it maps the
image space to itself, it is self-adjoint and positive semidefinite, and an
iterative solver applies it once per iteration. Its condition number is
the square of $A$'s, which is why the normal equations are solved iteratively
rather than by forming and factorizing anything.

**Conjugate gradients** ({class}`~bartorch.optim.CG`) solves them in a number
of iterations governed by the spectrum of $A^H A$, needing only its
application. With a full-rank $A$ and enough iterations it converges to the
least-squares solution — which, on undersampled data, is the wrong thing to
converge to: the null space is unconstrained, and the noise along the
poorly-determined directions is amplified by the inverse of their singular
values. Stopping the
iteration early is a form of regularization, and an unreliable one, since the
number of iterations that is right depends on the data.

## Regularization

Regularization adds to the data term a penalty that expresses what is expected
of the image:

$$
\hat{x} = \arg\min_x \; \tfrac12 \| A x - y \|_2^2 + \lambda R(x) .
$$

$R$ is the **regularization functional** or **prior**, and $\lambda > 0$ the
**regularization weight**, which sets the exchange rate between fitting the
data and satisfying the prior. Read as a maximum a posteriori estimate, the
data term is the negative log-likelihood of Gaussian noise and $\lambda R$ the
negative log-prior.

Two choices of $R$ cover most of what is used in MRI.

**Quadratic**, $R(x) = \|x\|_2^2$, is Tikhonov regularization. It keeps the
problem quadratic, so the solution still satisfies a linear system, now with
$A^H A + \lambda I$ in place of $A^H A$: better conditioned, solvable by
conjugate gradients, and biased toward small images. It suppresses noise
without exploiting any structure, and the result is a smoothed reconstruction.

**Nonsmooth and sparsity-promoting**, $R(x) = \|\Psi x\|_1$ for a transform
$\Psi$, states that the image has few large coefficients in $\Psi$ and many
negligible ones. Wavelet coefficients of an anatomical image are sparse in this
sense, as are the finite differences of a piecewise-smooth one — the latter
being total variation, $R(x) = \| \nabla x \|_1$. This is the prior that makes
compressed sensing work, and it is not differentiable, so the gradient methods
that suit the quadratic case do not apply.

## Proximal operators

The tool that handles a nonsmooth penalty is the **proximal operator** of a
functional $g$,

$$
\operatorname{prox}_{\tau g}(v) = \arg\min_u \; \tfrac{1}{2}\|u - v\|_2^2
    + \tau g(u) ,
$$

which takes a step toward decreasing $g$ without moving far from $v$. It is
the nonsmooth analogue of a gradient step, and for the penalties above it has
a closed form: the proximal operator of $\tau \|\cdot\|_1$ is soft
thresholding, $\operatorname{sign}(v) \max(|v| - \tau, 0)$, applied
coefficient by coefficient.

A term in {mod}`bartorch.priors` is exactly this pair: a functional's proximal
operator, and the linear transform $G$ it acts through, together representing
$g(Gx)$. {class}`~bartorch.priors.Wavelet` carries the wavelet transform
inside its proximal operator and has $G = I$;
{class}`~bartorch.priors.TotalVariation` has $G$ the finite-difference
operator, because the proximal operator of $\|\cdot\|_1$ composed with a
nontrivial $G$ has no closed form and the algorithm has to handle $G$ itself.
Which algorithms can is the next question.

## The algorithms

Every algorithm below splits the objective into the part it differentiates and
the part it takes proximal steps on, and each is a wrapper over BART's own
iteration.

**Iterative soft thresholding** ({class}`~bartorch.optim.IST`) alternates a
gradient step on the data term with the proximal operator of the penalty:

$$
x^{k+1} = \operatorname{prox}_{\tau \lambda R}
    \left( x^k - \tau A^H (A x^k - y) \right) .
$$

The step size $\tau$ is limited by the Lipschitz constant of the data term's
gradient, which is the largest eigenvalue of $A^H A$
({func}`~bartorch.optim.maxeigen` estimates it by power iteration).

**FISTA** ({class}`~bartorch.optim.FISTA`) is the same iteration with Nesterov
momentum, which improves the convergence rate of the objective from $O(1/k)$ to
$O(1/k^2)$ at no cost per iteration. It is the default choice for one penalty
whose transform is the identity.

**ADMM** ({class}`~bartorch.optim.ADMM`) splits the problem instead:
introducing $z = Gx$ and enforcing it with an augmented Lagrangian turns each
iteration into a least-squares solve in $x$ (by conjugate gradients, with
$A^H A + \rho G^H G$), a proximal step in $z$, and a dual update. It handles a
nontrivial $G$, and several penalties at once, at the cost of an inner solve
per iteration and a penalty parameter $\rho$ to choose.

**Primal-dual** ({class}`~bartorch.optim.PRIDU`) handles the same problems by
alternating a proximal step in the primal variable with one in the dual, using
$G$ and $G^H$ directly and never solving a linear system. It trades ADMM's
inner solve for more outer iterations.

The gradient methods are restricted to penalties whose transform is the
identity, because BART's `iter2_ist` takes a single term and ignores the
transform array: {class}`~bartorch.optim.IST` and
{class}`~bartorch.optim.FISTA` therefore refuse a term with a nontrivial $G$
rather than applying it as though $G$ were the identity.

## Data scaling and the regularization weight

The weight $\lambda$ is not dimensionless: multiplying the data by a constant
multiplies the data term by its square and leaves the penalty unchanged, so
the same $\lambda$ means something different on differently scaled data. BART's
reconstructions therefore normalize the data before they iterate, and
{func}`bartorch.optim.data_scaling` is that normalization — the estimate from
the k-space centre for a Cartesian acquisition, and from the spread of the
adjoint reconstruction $|A^H y|$ off the grid. A weight chosen on one dataset
transfers to another only because of this step.

## Where this lives in the library

{mod}`bartorch.linop` supplies $A$: an operator with a forward, an adjoint and
a normal, composable with `@` and `+`.  {mod}`bartorch.priors` supplies $R$ as
a proximal operator and its transform. {mod}`bartorch.optim` supplies the
algorithms, each called as `solver(y, A)`.
{func}`bartorch.tools.pics` assembles the three itself, inside BART's own
application: it estimates the scaling, builds the encoding from the coil
sensitivities and the sampling, turns its ``regularizers`` argument into
proximal operators, and hands all of it to the same iteration.

Two things follow from this arrangement. A reconstruction written as an
operator and a solver is the application's arithmetic rather than an
arithmetic that agrees with it — {doc}`../auto_examples/01-basics/02-operators-and-solvers`
checks that the two return the same bits. And an encoding BART has no
application for is written by composing operators, with the same solvers and
the same terms.

The estimators on this page all assume $A$ is known and linear. When the coil
sensitivities or a set of physical parameters are unknown too, it is not, and
{doc}`nonlinear` takes that case up.

## References

Lustig M, Donoho D, Pauly JM. Sparse MRI: the application of compressed sensing
for rapid MR imaging. *Magn Reson Med* 58(6):1182-1195 (2007).

Rudin LI, Osher S, Fatemi E. Nonlinear total variation based noise removal
algorithms. *Physica D* 60(1-4):259-268 (1992).

Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm for
linear inverse problems. *SIAM J Imaging Sci* 2(1):183-202 (2009).

Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. Distributed optimization and
statistical learning via the alternating direction method of multipliers.
*Found Trends Mach Learn* 3(1):1-122 (2011).

Chambolle A, Pock T. A first-order primal-dual algorithm for convex problems
with applications to imaging. *J Math Imaging Vis* 40(1):120-145 (2011).

Parikh N, Boyd S. Proximal algorithms. *Found Trends Optim* 1(3):127-239
(2014).
