# Applications

`bartorch.tools`: BART's applications, one function per command.  In each
section the hand-written wrappers come first; the rest are built from BART's
own declaration of the command and take its options under their long names.

```{eval-rst}
.. currentmodule:: bartorch.tools
```

## Simulation

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   phantom
   bloch
   coils
   epg
   fakeksp
   mobasig
   noise
   pulse
   seq
   signal
   sim
```

## Sampling and trajectories

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   traj
   bin
   estdelay
   estdims
   grid
   nufftbase
   pattern
   poisson
   psf
   raga
   rmfreq
   ssa
   trajcor
   upat
   wavepsf
```

## Coil calibration

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ecalib
   caldir
   calmat
   cc
   ccapply
   ecaltwo
   estscaling
   estvar
   ncalib
   phasepole
   rovir
   walsh
   whiten
```

## Reconstruction

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pics
   nlinv
   grog
   homodyne
   itsense
   looklocker
   lrmatrix
   moba
   mobafit
   pocsense
   rtnlinv
   sake
   sqpics
   wave
   wshfl
```
