:orphan:

API object index
================

Every documented object, by module.  The pages of the API reference list
the same objects in tables; this page generates the page of each.

bartorch
--------

.. currentmodule:: bartorch

.. autosummary::
   :toctree: generated
   :nosignatures:

   fft
   ifft
   fftshift
   fftmod
   nufft
   nufft_adjoint
   fwt
   iwt
   soft_thresh
   hard_thresh
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
   interpolate
   maps_to_kernels
   kernels_to_maps
   BartError
   set_num_threads
   set_copy_inputs
   set_debug_level
   get_debug_level
   cuda_available
   set_cuda_streams
   use_cuda_memcache
   bart_version
   build_info
   backend_sources

bartorch.linop
--------------

.. currentmodule:: bartorch.linop

.. autosummary::
   :toctree: generated
   :nosignatures:

   CartesianSense
   NoncartesianSense
   WaveSense
   FieldCorrected
   LinearOperator
   Identity
   Zero
   Diagonal
   ComponentDiagonal
   Conj
   Real
   FFT
   NUFFT
   MultiplySum
   Matrix
   Convolve
   Gradient
   concatenate
   stack
   hstack
   block_diag
   block
   Reshape
   Transpose
   Permute
   Flip
   Roll
   Pad
   Resize
   Extract
   Hankel
   Sum
   ScaledSum
   Mean
   Repeat

bartorch.nlop
-------------

.. currentmodule:: bartorch.nlop

.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearOperator
   TorchOperator
   NonlinearSense
   CartesianSense
   NoncartesianSense
   CoilSense
   SignalModel
   InversionRecovery
   MultiEcho
   Bloch
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
   SumOfSquares
   RootSumOfSquares
   IRGNM
   IRGNMBlock
   irgnm

bartorch.optim
--------------

.. currentmodule:: bartorch.optim

.. autosummary::
   :toctree: generated
   :nosignatures:

   CG
   IST
   FISTA
   ADMM
   PRIDU
   NIHT
   POCS
   Tikhonov
   maxeigen
   POCSBlock
   cg
   ist
   fista
   admm
   pridu
   niht
   pocs
   ISTBlock
   FISTABlock
   ADMMBlock
   PRIDUBlock
   FixedPoint
   data_scaling

bartorch.priors
---------------

.. currentmodule:: bartorch.priors

.. autosummary::
   :toctree: generated
   :nosignatures:

   L1
   Wavelet
   LocallyLowRank
   L2
   NonNegative
   TotalVariation
   FourierL1
   Laplace
   ImaginaryL1
   ImaginaryL2
   TotalGeneralizedVariation
   InfimalConvolutionTV
   InfimalConvolutionTGV
   WaveletNIHT
   ImageNIHT
   Regularizer
   ImplicitPrior
   frozen
   rof
   tgv
   nlmeans

bartorch.learning
-----------------

.. currentmodule:: bartorch.learning

.. autosummary::
   :toctree: generated
   :nosignatures:

   Denoiser
   Unrolled
   as_real
   as_complex

bartorch.apps
-------------

.. currentmodule:: bartorch.apps

.. autosummary::
   :toctree: generated
   :nosignatures:

   pics
   pocsense
   mobafit

bartorch.tools
--------------

.. currentmodule:: bartorch.tools

.. autosummary::
   :toctree: generated
   :nosignatures:

   phantom
   coils
   fakeksp
   noise
   traj
   grid
   pattern
   poisson
   upat
   raga
   psf
   wavepsf
   estdims
   estdelay
   trajcor
   rmfreq
   bin
   ssa
   ecalib
   caldir
   calmat
   ecaltwo
   walsh
   ncalib
   cc
   ccapply
   rovir
   whiten
   estvar
   estscaling
   phasepole
   pics
   nlinv
   rtnlinv
   moba
   mobafit
   looklocker
   itsense
   sqpics
   pocsense
   sake
   lrmatrix
   homodyne
   grog
   wave
   wshfl
   fovshift
   affine_transform
   warp
   register_affine
   register_nonrigid
   estimate_shift
   nrmse
   mse
   psnr
   ssim
   roi_stat

bartorch.io
-----------

.. currentmodule:: bartorch.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   readcfl
   writecfl

bartorch.cli
------------

.. currentmodule:: bartorch.cli

.. autosummary::
   :toctree: generated
   :nosignatures:

   main
   route
   read

bartorch.interop
----------------

.. currentmodule:: bartorch.interop

.. autosummary::
   :toctree: generated
   :nosignatures:

   to_deepinv

