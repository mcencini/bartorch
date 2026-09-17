# Array functions and settings

The functions of the `bartorch` namespace.  Shapes are C order, and an axis
argument is an index into a tensor's shape.

```{eval-rst}
.. currentmodule:: bartorch
```

## Fourier transforms

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   fft
   ifft
   fftshift
   fftmod
   nufft
   nufft_adjoint
```

## Wavelet transforms

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   fwt
   iwt
```

## Thresholding

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   soft_thresh
   hard_thresh
```

## Array utilities

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   resize
   flip
   circshift
   conv
   window
   median_filter
   moving_average
   normalize
   mip
   unwrap
   casorati
   rss
```

## Interpolation

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   interpolate
```

## Coil kernels

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   maps_to_kernels
   kernels_to_maps
```

## Errors

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   BartError
```

## Runtime settings

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   set_num_threads
   set_copy_inputs
   set_debug_level
   get_debug_level
```

## CUDA

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cuda_available
   set_cuda_streams
   use_cuda_memcache
```

## Build information

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   bart_version
   build_info
   backend_sources
```
