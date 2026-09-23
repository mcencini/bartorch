# BART commands as functions

`bartorch.tools` exposes BART's command-line applications as functions of
tensors, one function per command, run in this process.  Array arguments and
results are C-order tensors in BART's dimension order reversed, axis arguments
are axis indices rather than BART bitmasks, and regularization is given as
{mod}`bartorch.priors` terms rather than `-R` strings; results are not recorded
by autograd.  Functions marked *wrapped* have a hand-written signature; the
others are generated from BART's declaration of the command and take its
options under their long names.  {doc}`../explanation/execution-model`
compares these functions with the operators and solvers of the composable
interface.

```{eval-rst}
.. currentmodule:: bartorch.tools
```

## Simulation

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.phantom` | Analytical phantom as an image or as k-space, optionally with coils or along a trajectory (wrapped) |
| {obj}`~bartorch.tools.coils` | Analytical coil sensitivities |
| {obj}`~bartorch.tools.fakeksp` | K-space from an image and sensitivities |
| {obj}`~bartorch.tools.noise` | Additive complex Gaussian noise |

## Sampling and trajectories

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.traj` | Cartesian, radial, golden-angle and spiral trajectories in grid units (wrapped) |
| {obj}`~bartorch.tools.grid` | Sampling grid coordinates in image space or k-space |
| {obj}`~bartorch.tools.pattern` | Sampling pattern of a k-space array |
| {obj}`~bartorch.tools.poisson` | Poisson-disc sampling pattern |
| {obj}`~bartorch.tools.upat` | Regular undersampling pattern with a fully sampled centre |
| {obj}`~bartorch.tools.raga` | Indices of a RAGA (rational approximation of the golden angle) radial ordering |
| {obj}`~bartorch.tools.psf` | Point spread function of a trajectory |
| {obj}`~bartorch.tools.wavepsf` | Wave-CAIPI point spread function in hybrid space |
| {obj}`~bartorch.tools.estdims` | Image dimensions implied by a non-Cartesian trajectory |
| {obj}`~bartorch.tools.estdelay` | Gradient-delay estimation from radial data |
| {obj}`~bartorch.tools.trajcor` | Gradient-delay correction of a trajectory |
| {obj}`~bartorch.tools.rmfreq` | Removal of an angle-dependent frequency from radial data |
| {obj}`~bartorch.tools.bin` | Binning of data by labels, including respiratory and cardiac quadrature binning |
| {obj}`~bartorch.tools.ssa` | Singular spectrum analysis (SSA-FARY) of a time series |

## Coil calibration and compression

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.ecalib` | Sensitivities by ESPIRiT (wrapped) |
| {obj}`~bartorch.tools.caldir` | Sensitivities directly from the low-resolution k-space centre (wrapped) |
| {obj}`~bartorch.tools.calmat` | Calibration matrix of the k-space centre |
| {obj}`~bartorch.tools.ecaltwo` | Second stage of ESPIRiT calibration |
| {obj}`~bartorch.tools.walsh` | Walsh coil combination, for use with {obj}`~bartorch.tools.ecaltwo` |
| {obj}`~bartorch.tools.ncalib` | Sensitivities from non-Cartesian data by nonlinear inversion (ENLIVE) |
| {obj}`~bartorch.tools.cc` | Coil compression matrix (SVD, geometric or ESPIRiT) |
| {obj}`~bartorch.tools.ccapply` | Application of a coil compression matrix |
| {obj}`~bartorch.tools.rovir` | Coil compression by region-optimized virtual coils (ROVir) |
| {obj}`~bartorch.tools.whiten` | Noise prewhitening from a noise measurement |
| {obj}`~bartorch.tools.estvar` | Noise variance of white Gaussian noise |
| {obj}`~bartorch.tools.estscaling` | Data scaling estimated from the k-space centre |
| {obj}`~bartorch.tools.phasepole` | Detection of phase poles in sensitivities |

## Reconstruction

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.pics` | Parallel-imaging compressed-sensing reconstruction (wrapped) |
| {obj}`~bartorch.tools.nlinv` | Nonlinear inversion: image and sensitivities jointly (wrapped) |
| {obj}`~bartorch.tools.rtnlinv` | Real-time nonlinear inversion of a time series |
| {obj}`~bartorch.tools.moba` | Model-based nonlinear reconstruction of parameter maps |
| {obj}`~bartorch.tools.mobafit` | Voxel-wise fit of a signal model to contrast images |
| {obj}`~bartorch.tools.looklocker` | $T_1$ from Look-Locker parameters $M_0$, $M_{ss}$ and $R_1^*$ |
| {obj}`~bartorch.tools.itsense` | Iterative SENSE with $\ell_2$ regularization |
| {obj}`~bartorch.tools.sqpics` | Parallel-imaging compressed-sensing reconstruction, BART's `sqpics` variant of `pics` |
| {obj}`~bartorch.tools.pocsense` | POCSENSE reconstruction |
| {obj}`~bartorch.tools.sake` | SAKE: low-rank matrix completion of k-space |
| {obj}`~bartorch.tools.lrmatrix` | Multi-scale low-rank matrix completion |
| {obj}`~bartorch.tools.homodyne` | Homodyne partial-Fourier reconstruction |
| {obj}`~bartorch.tools.grog` | GROG calibration and gridding of radial data |
| {obj}`~bartorch.tools.wave` | Wave-CAIPI reconstruction |
| {obj}`~bartorch.tools.wshfl` | Wave-Shuffling reconstruction |

## Preprocessing, registration and metrics

| Object | Description |
| --- | --- |
| {obj}`~bartorch.tools.fovshift` | Field-of-view shift of k-space by a linear phase (wrapped) |
| {obj}`~bartorch.tools.affine_transform` | Resampling on an affinely mapped grid (wrapped) |
| {obj}`~bartorch.tools.warp` | Resampling through a displacement field (wrapped) |
| {obj}`~bartorch.tools.register_affine` | Affine registration by mutual information (wrapped) |
| {obj}`~bartorch.tools.register_nonrigid` | Non-rigid registration by greedy SyN or TV-L1 optical flow (wrapped) |
| {obj}`~bartorch.tools.estimate_shift` | Sub-voxel translation from the phase of the cross-spectrum (wrapped) |
| {obj}`~bartorch.tools.nrmse` | Normalized root-mean-square error, optionally after least-squares scaling (wrapped) |
| {obj}`~bartorch.tools.mse` | Mean squared error (wrapped) |
| {obj}`~bartorch.tools.psnr` | Peak signal-to-noise ratio of the magnitudes (wrapped) |
| {obj}`~bartorch.tools.ssim` | Structural similarity of the magnitudes (wrapped) |
| {obj}`~bartorch.tools.roi_stat` | Statistic over a region of interest (wrapped) |
