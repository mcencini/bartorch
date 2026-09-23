# Array functions and settings

The `bartorch` namespace holds array functions that call BART's commands on
tensors, and the process-wide settings of the embedded library.  Axis
arguments are indices into a C-order shape; results are not recorded by
autograd.

```{eval-rst}
.. currentmodule:: bartorch
```

## Fourier transforms

| Object | Description |
| --- | --- |
| {obj}`~bartorch.fft` | Fourier transform along axes; centred and unnormalized by default, unitary with `unitary=True` |
| {obj}`~bartorch.ifft` | Inverse of {obj}`~bartorch.fft` |
| {obj}`~bartorch.fftshift` | Cyclic shift of the zero frequency to the centre, or back |
| {obj}`~bartorch.fftmod` | Multiplication by the alternating phase relating centred and uncentred transforms |
| {obj}`~bartorch.nufft` | Non-uniform Fourier transform of an image along a trajectory |
| {obj}`~bartorch.nufft_adjoint` | Adjoint non-uniform Fourier transform, from samples to an image |

## Wavelet transforms

| Object | Description |
| --- | --- |
| {obj}`~bartorch.fwt` | Multi-level discrete wavelet transform along axes |
| {obj}`~bartorch.iwt` | Inverse of {obj}`~bartorch.fwt` |

## Thresholding

| Object | Description |
| --- | --- |
| {obj}`~bartorch.soft_thresh` | Soft thresholding of the modulus, $x \max(1 - \lambda/\lvert x\rvert, 0)$ |
| {obj}`~bartorch.hard_thresh` | Hard thresholding, keeping entries with $\lvert x\rvert > \lambda$ |

## Array utilities

| Object | Description |
| --- | --- |
| {obj}`~bartorch.resize` | Cropping or zero padding to a shape, anchored at the centre, the front or the start |
| {obj}`~bartorch.flip` | Reversal along axes |
| {obj}`~bartorch.circshift` | Cyclic shift along axes |
| {obj}`~bartorch.conv` | Cyclic convolution with a kernel |
| {obj}`~bartorch.window` | Hamming or Hann window along axes |
| {obj}`~bartorch.median_filter` | Moving median along one axis |
| {obj}`~bartorch.moving_average` | Moving average along one axis |
| {obj}`~bartorch.normalize` | Division by the $\ell_2$ or $\ell_1$ norm over axes |
| {obj}`~bartorch.mip` | Maximum or minimum intensity projection |
| {obj}`~bartorch.unwrap` | Unwrapping of the real part along one axis, for a given period |
| {obj}`~bartorch.casorati` | Casorati matrix of overlapping blocks |
| {obj}`~bartorch.rss` | Root sum of squares over axes |

## Interpolation

| Object | Description |
| --- | --- |
| {obj}`~bartorch.interpolate` | Sampling of an array at fractional voxel positions |

## Coil kernels

| Object | Description |
| --- | --- |
| {obj}`~bartorch.maps_to_kernels` | Sensitivities as their low-frequency k-space kernels |
| {obj}`~bartorch.kernels_to_maps` | Sensitivities from k-space kernels |

## Errors

| Object | Description |
| --- | --- |
| {obj}`~bartorch.BartError` | Exception raised when a BART command or operator fails, carrying BART's message |

## Runtime settings

| Object | Description |
| --- | --- |
| {obj}`~bartorch.set_num_threads` | Threads used by BART, its FFT and FINUFFT's host transforms |
| {obj}`~bartorch.set_copy_inputs` | Whether commands work on copies of their inputs (default) |
| {obj}`~bartorch.set_debug_level` | BART's verbosity |
| {obj}`~bartorch.get_debug_level` | Current verbosity |

## CUDA

| Object | Description |
| --- | --- |
| {obj}`~bartorch.cuda_available` | Whether the library was built with CUDA and a device is present |
| {obj}`~bartorch.set_cuda_streams` | Number of CUDA streams BART uses |
| {obj}`~bartorch.use_cuda_memcache` | Whether BART caches freed device memory |

## Build information

| Object | Description |
| --- | --- |
| {obj}`~bartorch.bart_version` | Version of the embedded BART |
| {obj}`~bartorch.build_info` | Compiler, nested-function mode, OpenMP and CUDA configuration of the library |
| {obj}`~bartorch.backend_sources` | Library serving each BLAS and LAPACK routine and the FFT |
