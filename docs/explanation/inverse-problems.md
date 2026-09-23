# Inverse problems and their solvers

An MRI reconstruction estimates an image from measurements that determine it
incompletely or unstably.  This page states the estimation problem, the
properties of the forward operator that make it difficult, the regularized
formulations used in its place, and the algorithms that solve them.  The
organization follows the operator–functional–algorithm structure of the Pyxu
documentation,[^pyxu] specialized to MRI.

## The measurement model

The measured data $y \in \mathbb{C}^M$ and the unknown image
$x \in \mathbb{C}^N$ are related by

$$
y = A x + \varepsilon ,
$$

with $A : \mathbb{C}^N \to \mathbb{C}^M$ the **forward operator** — in MRI the
**encoding operator**, the subject of {doc}`encoding` — and $\varepsilon$ the
noise.  After prewhitening, the noise of MRI data is modelled as complex
Gaussian with independent, identically distributed entries.  For fixed coil
sensitivities $A$ is linear, and it is applied without being formed as a
matrix: for a $256^2$ image and eight coils the matrix would have about
$4\times 10^9$ entries, while an application costs eight FFTs.

The **adjoint** $A^H$ is defined by
$\langle Ax, y\rangle = \langle x, A^H y\rangle$ for all $x$ and $y$; for a
matrix it is the conjugate transpose.  The **normal operator** $A^H A$ maps the
image space to itself and is self-adjoint and positive semidefinite.  Every
algorithm on this page needs only applications of $A$ and $A^H$, or of
$A^H A$, so one solver serves every encoding.  $A^H y$ is the simplest
estimate; for Cartesian SENSE it is the coil combination of the zero-filled
inverse FFTs, and it retains the aliasing of the undersampling.

## Least squares

The least-squares estimate minimizes the data misfit,

$$
\hat{x} \in \arg\min_x \; \tfrac12 \lVert Ax - y \rVert_2^2 ,
$$

whose minimizers are the solutions of the **normal equations**
$A^H A \hat{x} = A^H y$.  With the singular value decomposition
$A = \sum_i \sigma_i u_i v_i^H$, the minimizer of least norm is

$$
x^\dagger = \sum_{\sigma_i > 0} \frac{u_i^H y}{\sigma_i} \, v_i .
$$

Two properties of $A$ decide whether $x^\dagger$ is useful.

| Property | Condition | Consequence |
| --- | --- | --- |
| Rank deficiency | $A$ has a nontrivial null space: some $x_0 \ne 0$ satisfy $Ax_0 = 0$ | The minimizer is not unique: $x^\dagger + x_0$ fits the data equally well, and the data carry no information about $x_0$ |
| Ill-conditioning | $A$ has full column rank, but its smallest singular values are small | The minimizer is unique, but the noise component along $v_i$ is amplified by $1/\sigma_i$ |

In parallel imaging, an undersampled acquisition measures fewer k-space
positions than the image has voxels, but each position is measured by several
coils.  If the coils' sensitivities separate the voxels that alias onto one
another, $A$ has full column rank and the problem is ill-conditioned rather
than underdetermined; the spatially resolved noise amplification is the
g-factor of SENSE.[^pruessmann1999]  At accelerations beyond what the coil
geometry supports, and outside the region the coils cover, $A$ becomes rank
deficient.

The ratio $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ of the largest to the
smallest nonzero singular value is the spectral condition number.  The
eigenvalues of $A^H A$ are the $\sigma_i^2$, so
$\kappa(A^H A) = \kappa(A)^2$ for any $A$, with $\sigma_{\min}$ the smallest
nonzero singular value when $A$ is rank deficient.

## Conjugate gradients and early stopping

**Conjugate gradients** applied to the normal equations
({class}`~bartorch.optim.CG`) needs only $A^H A$.  In exact arithmetic its
iterates started from $x_0 = 0$ remain in the range of $A^H$, the orthogonal
complement of the null space, so they converge to the minimum-norm solution
$x^\dagger$; a nonzero $x_0$ keeps its null-space component unchanged.  The
error decreases at a rate governed by $\kappa(A^H A)$.

On noisy data the iteration shows semi-convergence: the early iterates
recover the components with large $\sigma_i$, and the later ones increasingly
fit the noise amplified along the components with small $\sigma_i$.  The
iteration count therefore acts as a regularization parameter,[^hansen] as in
iterative SENSE,[^pruessmann2001] but its best value depends on the data and
the noise level, and no reconstruction property is attached to a particular
count.

## Regularized estimation

Regularization adds a functional expressing prior knowledge of the image,

$$
\hat{x} = \arg\min_x \; \tfrac12 \lVert Ax - y \rVert_2^2 + \lambda R(x),
\qquad \lambda > 0 .
$$

A **functional** maps an image to a real number.  $R$ is the **regularization
functional** and $\lambda$ the **regularization weight**.  In a Bayesian
reading, the data term is the negative log-likelihood of white Gaussian noise
and $\lambda R$ the negative logarithm of a prior density, so $\hat{x}$ is a
maximum a posteriori estimate.

| Regularization | $R(x)$ | Effect |
| --- | --- | --- |
| Tikhonov | $\tfrac12 \lVert x \rVert_2^2$ | Solution $(A^H A + \lambda I)^{-1} A^H y$: the component along $v_i$ is weighted by $\sigma_i^2 / (\sigma_i^2 + \lambda)$, so poorly determined components are shrunk toward zero.  It is not a spatial smoothness prior. |
| Generalized Tikhonov | $\tfrac12 \lVert Gx - b \rVert_2^2$ | Shrinkage toward a reference $b$, or with $G$ a finite-difference operator, a quadratic smoothness prior |
| Sparsity in a transform | $\lVert \Psi x \rVert_1$ | Few significant coefficients, for example of a wavelet transform $\Psi$; the prior of compressed sensing[^lustig] |
| Total variation | $\sum_r \lVert (\nabla x)_r \rVert_2$ | Piecewise-constant images[^rof] |
| Locally low rank | $\sum_b \lVert B_b x \rVert_*$ | Image blocks, arranged as matrices over a contrast or coefficient axis, of low rank |

The quadratic functionals keep the problem a linear least-squares problem.
The others are not differentiable, and require the methods below.

## Differentiable and proximable terms

The data term $f(x) = \tfrac12\lVert Ax - y\rVert_2^2$ is differentiable, with
gradient $\nabla f(x) = A^H(Ax - y)$.  Its gradient is Lipschitz continuous
with constant $L = \lVert A \rVert^2 = \lambda_{\max}(A^H A)$, which bounds
the step size of a gradient method.

A nonsmooth functional $g$ is used through its **proximal operator**

$$
\operatorname{prox}_{\tau g}(v) = \arg\min_u \; \tfrac12 \lVert u - v \rVert_2^2 + \tau g(u),
$$

which is defined for any proper, closed, convex $g$ and $\tau > 0$.  For the
$\ell_1$ norm of a complex array it is soft thresholding of the modulus, which
preserves the phase:

$$
\operatorname{prox}_{\tau \lVert\cdot\rVert_1}(v)_i =
\frac{v_i}{\lvert v_i \rvert} \max\!\left(\lvert v_i \rvert - \tau,\, 0\right),
$$

with value zero where $v_i = 0$.  Thresholding groups of entries jointly — the
directions of a gradient in isotropic total variation, or the axes a term's
`joint_axes` name — replaces $\lvert v_i \rvert$ by the $\ell_2$ norm of the
group.  The proximal operator of the indicator function of a convex set is the
projection onto the set.

A regularization term of {mod}`bartorch.priors` represents $g(Gx)$: the
proximal operator of $g$, built by BART, and a linear transform $G$.  When $G$
is unitary, $\operatorname{prox}_{\tau g \circ G}(v) = G^H \operatorname{prox}_{\tau g}(Gv)$,
and the transform can be placed inside the proximal operator.  BART's wavelet
and locally low-rank terms apply their transforms inside the proximal operator
in this way and have $G = I$.  The relation is exact for an orthogonal wavelet
(Haar, Daubechies) and for non-overlapping blocks; with the biorthogonal
CDF 4/4 wavelet, with overlapping blocks, and with the random shifts between
iterations that both terms apply by default, the step is an approximation of
that proximal operator.  For a general $G$, such as the finite differences of
total variation, $\operatorname{prox}_{\tau g\circ G}$ has no closed form, and
the algorithm has to treat $G$ separately.

## Algorithms

| Algorithm | Problem and splitting | Transform $G$ | Per iteration | Convergence conditions |
| --- | --- | --- | --- | --- |
| CG | Quadratic: $\min \lVert Ax - y\rVert^2 + \lambda\lVert x\rVert^2 + \sum_i w_i\lVert G_i x - b_i\rVert^2$ | Any, inside the quadratic terms | One application of the normal operator | Positive semidefinite normal operator |
| IST (proximal gradient) | $f + g$: $x \leftarrow \operatorname{prox}_{\tau\lambda g}\!\left(x - \tau \nabla f(x)\right)$ | $I$ only | One normal operator, one proximal operator | Convex $f$ and $g$, $0 < \tau \le 1/L$; objective error $O(1/k)$ |
| FISTA | As IST, with Nesterov momentum | $I$ only | As IST | As IST; objective error $O(1/k^2)$[^beck] |
| ADMM | $f(x) + \sum_j g_j(z_j)$ subject to $z_j = G_j x$ | Any | A CG solve with $A^H A + \rho \sum_j G_j^H G_j$, one proximal operator per term, a dual update | Convex terms, any $\rho > 0$[^boyd] |
| Primal-dual (Chambolle–Pock) | Saddle-point form with a dual variable for each $g_j(G_j x)$ | Any | Applications of $A$, $G_j$ and their adjoints, proximal operators; no inner solve | Convex terms, $\sigma\tau\lVert K\rVert^2 < 1$ for the stacked operator $K$[^chambolle] |
| POCS | Feasibility: repeated projection onto convex sets | Inside the projections | One projection per set | Closed convex sets with nonempty intersection |

**Proximal gradient and FISTA.**  Each iteration takes a gradient step on the
data term and a proximal step on the regularization.  The step size must not
exceed $1/L$.  BART's default step size, used by {class}`~bartorch.optim.IST`,
{class}`~bartorch.optim.FISTA` and {class}`~bartorch.optim.PRIDU`, is $0.95$ in
absolute units.  This satisfies the bound for Cartesian SENSE with a unitary
FFT, a binary sampling pattern and sensitivities normalized to unit root sum
of squares over the coils, where $\lVert A \rVert \le 1$.  For other encodings,
non-Cartesian ones in particular, `eigen=True` divides the step by
$\lambda_{\max}(A^H A)$, estimated by 30 power iterations as
{func}`~bartorch.optim.maxeigen` does.  BART's IST and FISTA apply a single
proximal operator to the image, so these solvers accept one term with
$G = I$ and raise an error for any other.

**ADMM** introduces $z_j = G_j x$ and alternates a linear least-squares
update of $x$, solved by conjugate gradients, a proximal step for each $z_j$,
and an update of the scaled dual variables.  It accepts several terms and any
$G_j$, at the cost of an inner solve per iteration and a penalty parameter
$\rho$.

**The primal-dual method** alternates proximal steps on the primal and the
dual variables and applies $G_j$ and $G_j^H$ directly, with no inner solve; it
usually needs more iterations than ADMM for the same accuracy.

## Data scaling and the regularization weight

$\lambda$ has units.  Multiplying the data by $c$ multiplies the data term by
$c^2$: for a term homogeneous of degree one, such as an $\ell_1$ norm or total
variation, the minimizer then scales by $c$ only if $\lambda$ scales by $c$ as
well.  BART's reconstructions therefore divide the data by an estimate of its
scale before iterating, and a weight is chosen for normalized data.
{func}`bartorch.optim.data_scaling` is that estimate.  For a Cartesian
acquisition it is computed from a central calibration region of k-space: the
root-sum-of-squares image of that region, corrected for its size, and the 90th
percentile of its voxel magnitudes — or their maximum, when the maximum exceeds
the 90th percentile by at least twice the difference between the 90th
percentile and the median.  For a non-Cartesian acquisition the
same statistic is taken of $\lvert A^H y \rvert$.  `pics` returns the
reconstruction of the scaled data without scaling it back.

## Representation in bartorch

| Object | bartorch |
| --- | --- |
| $A$, $A^H$, $A^H A$ | {class}`~bartorch.linop.LinearOperator`: `A(x)`, `A.H(y)`, `A.normal(x)` |
| $\lambda R$, $g(Gx)$ | A {mod}`bartorch.priors` term, or an {class}`~bartorch.priors.ImplicitPrior` |
| Quadratic terms | {class}`~bartorch.optim.Tikhonov` for {class}`~bartorch.optim.CG` |
| Algorithm | A solver of {mod}`bartorch.optim`, called as `solver(y, A)` |
| One iteration | An iteration block: `start`, `forward`, `output` |
| $L$ | {func}`~bartorch.optim.maxeigen` |
| Data scale | {func}`~bartorch.optim.data_scaling` |

{func}`bartorch.tools.pics` performs these steps inside BART: it estimates the
data scale, builds the encoding from the sensitivities and the sampling,
builds the terms from its `regularizers` argument, and runs the chosen
iteration.  {doc}`../auto_examples/01-basics/02-operators-and-solvers`
assembles the same reconstruction from an operator, a term and a solver and
obtains the same result.  The estimators on this page assume a known, linear
$A$; {doc}`nonlinear` treats forward operators that depend nonlinearly on the
unknowns.

## References

[^pyxu]: Pyxu developers. Pyxu documentation, <https://pyxu-org.github.io/>: the guides on forward operators, functionals, operator algebra, optimization algorithms and Lipschitz constants.

[^pruessmann1999]: Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity encoding for fast MRI. *Magn Reson Med* 42(5):952–962 (1999). [doi:10.1002/(SICI)1522-2594(199911)42:5\<952::AID-MRM16\>3.0.CO;2-S](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S)

[^pruessmann2001]: Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in sensitivity encoding with arbitrary k-space trajectories. *Magn Reson Med* 46(4):638–651 (2001). [doi:10.1002/mrm.1241](https://doi.org/10.1002/mrm.1241)

[^hansen]: Hansen PC. *Rank-Deficient and Discrete Ill-Posed Problems: Numerical Aspects of Linear Inversion.* SIAM (1998). [doi:10.1137/1.9780898719697](https://doi.org/10.1137/1.9780898719697)

[^lustig]: Lustig M, Donoho D, Pauly JM. Sparse MRI: the application of compressed sensing for rapid MR imaging. *Magn Reson Med* 58(6):1182–1195 (2007). [doi:10.1002/mrm.21391](https://doi.org/10.1002/mrm.21391)

[^rof]: Rudin LI, Osher S, Fatemi E. Nonlinear total variation based noise removal algorithms. *Physica D* 60(1–4):259–268 (1992). [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F)

[^beck]: Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM J Imaging Sci* 2(1):183–202 (2009). [doi:10.1137/080716542](https://doi.org/10.1137/080716542)

[^boyd]: Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. Distributed optimization and statistical learning via the alternating direction method of multipliers. *Found Trends Mach Learn* 3(1):1–122 (2011). [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)

[^chambolle]: Chambolle A, Pock T. A first-order primal-dual algorithm for convex problems with applications to imaging. *J Math Imaging Vis* 40(1):120–145 (2011). [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)

Further reading: Parikh N, Boyd S. Proximal algorithms. *Found Trends Optim* 1(3):127–239 (2014). [doi:10.1561/2400000003](https://doi.org/10.1561/2400000003)
