# Sampling off the Cartesian grid

A radial, spiral or other non-Cartesian acquisition samples k-space at
positions that are not on a grid, so its Fourier transform is not an FFT.  This
page defines the transform bartorch computes, its accuracy and its conventions,
the difference between the adjoint, the density-compensated adjoint and a
reconstruction, the normal operator used by iterative solvers, and which
backend computes what.

## The transform

For an image $x$ of $N$ voxels and $M$ sample positions $k_j$, the forward
transform is

$$
y_j = \frac{1}{\sqrt{N}} \sum_{m} x_m \, \exp\!\left(-2\pi i \sum_d \frac{k_{j,d}\, m_d}{n_d}\right),
\qquad j = 1, \dots, M,
$$

with $m$ the voxel index relative to the image centre
($m_d = -\lfloor n_d/2 \rfloor, \dots$ along each dimension $d$ of size $n_d$)
and $k_j$ in **grid units**: multiples of $1/\mathrm{FOV}$, so that a fully
sampled readout of $n$ samples spans $-n/2$ to $n/2$.  The adjoint has the
positive exponent and the same $1/\sqrt{N}$ scaling.  A trajectory always has
three components; a $k_z$ that is zero throughout makes the transform
two-dimensional.

Evaluated directly the sum costs $O(NM)$.  A **non-uniform fast Fourier
transform** (NUFFT) computes it to a prescribed accuracy in
$O(N \log N + M)$: the samples are spread onto a grid oversampled by a factor
$\sigma$ with a compact kernel, the grid is transformed by an FFT, and the
result is divided by the kernel's Fourier transform (**deapodization**); the
adjoint performs the same steps in reverse.[^osullivan][^fessler2003][^beatty]
The kernel width and $\sigma$ determine the accuracy.

In bartorch the transform is computed by FINUFFT,[^finufft] whose forward
operator is its type-2 transform and whose adjoint is its type-1 transform.
FINUFFT is planned with a relative **tolerance** $\varepsilon$, from which it
chooses its kernel width.

| Setting | Default | Set by |
| --- | --- | --- |
| Tolerance $\varepsilon$ | $10^{-3}$; $10^{-6}$ for `ncalib`, `nlinv` and `rtnlinv` | The kernel width: `width` of {class}`~bartorch.linop.NUFFT`, BART's `-w` option of the commands, converted to the tolerance that gives that width |
| Oversampling $\sigma$ | 1.25; 2 for `ncalib`, `nlinv` and `rtnlinv` | `oversampling` of {class}`~bartorch.linop.NUFFT`, BART's `-o` option |

The relative error of a transform is of the order of $\varepsilon$; the test
suite compares it with the explicit sum above.  The default trades accuracy
for speed and memory, and is intended for reconstruction, where the error of
the transform is meant to stay small beside the errors caused by noise and
undersampling.  A comparison with an analytical reference or between
implementations needs a larger kernel width.  cuFINUFFT accepts $\sigma = 2$ and $\sigma = 1.25$ only.

## Adjoint, density compensation and reconstruction

| Estimate | Definition | Property |
| --- | --- | --- |
| Adjoint | $\mathrm{NUFFT}^H y$ | Each sample is added onto the grid; regions sampled more densely are weighted more.  For radial sampling, which samples the k-space centre once per spoke, the result is the image convolved with a strongly low-pass kernel |
| Density-compensated adjoint (gridding reconstruction) | $\mathrm{NUFFT}^H D y$, with $D$ a diagonal of weights $d_j \approx 1/\rho(k_j)$ | Approximately equalizes the sampling density $\rho$; for radial sampling $d_j \propto \lvert k_j \rvert$.  Unsampled regions of k-space remain missing, and undersampling appears as streaks |
| Reconstruction | Solution of the least-squares or regularized problem with $A$ | Consistent with the measured samples; needs no density compensation, which can serve as a weighting of the data term instead[^pipe][^pruessmann2001] |

Density weights passed to {class}`~bartorch.linop.NUFFT` or
{class}`~bartorch.linop.NoncartesianSense` become part of the operator,
$A = W\,\mathrm{NUFFT}\,S$: the weights are applied on the forward pass and
their conjugate on the adjoint, and the normal operator is built over them.
Weights that do not lie along the k-space samples are refused.

## The normal operator and the point spread function

An iterative solver applies $A^H A$.  The transform's own normal operator,
$T = \mathrm{NUFFT}^H W^H W\,\mathrm{NUFFT}$, is a convolution of the image with
the **point spread function** (PSF)

$$
h(m) = \frac{1}{N} \sum_j \lvert w_j \rvert^2 \exp\!\left(2\pi i \sum_d \frac{k_{j,d}\, m_d}{n_d}\right),
$$

whose support spans twice the image extent.  $T$ is therefore applied exactly
as a multiplication, in the Fourier domain of a grid doubled in each dimension,
by the **transfer function** $\hat{h}$, the FFT of $h$ on that grid: the image
is zero-padded to the doubled grid, transformed, multiplied, transformed back
and cropped.[^fessler2005]  With a subspace basis $h$ becomes a set of kernels
over pairs of coefficients.  The full SENSE normal operator
$\sum_c \overline{S_c}\, T\, S_c$ is not a convolution, because the
sensitivities vary in space; {doc}`encoding` describes how it is applied.

$h$ is computed as the adjoint NUFFT of $\lvert w_j \rvert^2$ (of ones without
weights) onto the doubled grid, by FINUFFT, and BART stores $\hat{h}$ and
performs the multiplication.  Its storage options remain available:

| BART `--nufft-conf` | Storage of $\hat{h}$ |
| --- | --- |
| default, `lowmem`, `no-precomp` | The complete complex array |
| `decomposed-psf` | $2^d$ grids of the image size, one at a time |
| `upper-triag-psf` | Half of the Hermitian kernel matrix of a subspace basis |
| `real-psf` | The real part |
| `compress-psf` | The entries within the trajectory's footprint, and their indices |

One application of this normal operator costs two FFTs of the doubled grid per
coil and one multiplication; the alternative, a forward and an adjoint NUFFT,
spreads and interpolates every sample of every coil.  `toeplitz=False` on an
operator, or BART's `pics --no-toeplitz`, selects the two transforms instead.
The two forms differ by an amount of the order of the transform's tolerance,
as {doc}`../auto_examples/02-non-cartesian/01-trajectories-and-transforms`
measures.

## Backends and refusals

Every non-uniform transform is computed by FINUFFT or cuFINUFFT; BART's own
gridding implementation is not used, and no configuration falls back to it.

| Path | Backend | Device | Requirement and behaviour when unavailable |
| --- | --- | --- | --- |
| Forward and adjoint transform, PSF | FINUFFT | CPU | Installed as a dependency; without it, a non-uniform transform raises an error |
| Forward and adjoint transform, PSF | cuFINUFFT | CUDA | The `cufinufft` extra.  On a machine with a CUDA device and a CUDA build of bartorch, the substitution is not installed without it, and every non-uniform transform, on the host as well, raises an error |
| Multiplication by $\hat h$ | BART, with the FFT of the device: MKL or pocketfft on the host, cuFFT on a CUDA device | Either | — |
| Unsupported configuration | None | Either | {class}`~bartorch.BartError` naming the reason: weights that do not lie along k-space, an image that varies along an axis the trajectory indexes, a kernel width no tolerance produces, among others |

The backend of a transform is chosen by where its arguments are, not by where
the trajectory is: an operator holds one pair of plans per memory space and
builds each the first time a transform is requested there.  On macOS, the
OpenMP runtimes of the PyTorch and FINUFFT wheels must first be made one, as
{ref}`the installation guide <macos-openmp>` describes.

## Trajectories

{func}`bartorch.tools.traj` generates trajectories in grid units.  A radial
trajectory with successive spokes separated by $\pi/n_s$ covers k-space
uniformly for a frame of exactly $n_s$ spokes.  With **golden-angle**
ordering, successive spokes are separated by $\pi$ times the reciprocal of
the golden ratio, about $111.25°$, and any number of consecutive spokes covers
k-space approximately uniformly.[^winkelmann]  A continuously acquired
golden-angle series can therefore be divided into frames after the
acquisition, as {doc}`../auto_examples/03-applications/01-dynamic-golden-angle`
does.

## References

[^osullivan]: O'Sullivan JD. A fast sinc function gridding algorithm for Fourier inversion in computer tomography. *IEEE Trans Med Imaging* 4(4):200–207 (1985). [doi:10.1109/TMI.1985.4307723](https://doi.org/10.1109/TMI.1985.4307723)

[^fessler2003]: Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. *IEEE Trans Signal Process* 51(2):560–574 (2003). [doi:10.1109/TSP.2002.807005](https://doi.org/10.1109/TSP.2002.807005)

[^beatty]: Beatty PJ, Nishimura DG, Pauly JM. Rapid gridding reconstruction with a minimal oversampling ratio. *IEEE Trans Med Imaging* 24(6):799–808 (2005). [doi:10.1109/TMI.2005.848376](https://doi.org/10.1109/TMI.2005.848376)

[^finufft]: Barnett AH, Magland J, af Klinteberg L. A parallel nonuniform fast Fourier transform library based on an "exponential of semicircle" kernel. *SIAM J Sci Comput* 41(5):C479–C504 (2019). [doi:10.1137/18M120885X](https://doi.org/10.1137/18M120885X).  For the CUDA library: Shih Y, Wright G, Andén J, Blaschke J, Barnett AH. cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. *IEEE IPDPSW* 688–697 (2021). [doi:10.1109/IPDPSW52791.2021.00105](https://doi.org/10.1109/IPDPSW52791.2021.00105)

[^pipe]: Pipe JG, Menon P. Sampling density compensation in MRI: rationale and an iterative numerical solution. *Magn Reson Med* 41(1):179–186 (1999). [doi:10.1002/(SICI)1522-2594(199901)41:1\<179::AID-MRM25\>3.0.CO;2-V](https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V)

[^pruessmann2001]: Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in sensitivity encoding with arbitrary k-space trajectories. *Magn Reson Med* 46(4):638–651 (2001). [doi:10.1002/mrm.1241](https://doi.org/10.1002/mrm.1241)

[^fessler2005]: Fessler JA, Lee S, Olafsson VT, Shi HR, Noll DC. Toeplitz-based iterative image reconstruction for MRI with correction for magnetic field inhomogeneity. *IEEE Trans Signal Process* 53(9):3393–3402 (2005). [doi:10.1109/TSP.2005.853152](https://doi.org/10.1109/TSP.2005.853152)

[^winkelmann]: Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal radial profile order based on the Golden Ratio for time-resolved MRI. *IEEE Trans Med Imaging* 26(1):68–76 (2007). [doi:10.1109/TMI.2006.885337](https://doi.org/10.1109/TMI.2006.885337)
