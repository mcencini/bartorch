# Related projects

## Components of bartorch

| Project | Relationship |
| --- | --- |
| [BART](https://mrirecon.codeberg.page/) | The reconstruction toolbox bartorch embeds, pinned as the `external/bart` submodule and compiled without source changes |
| [PyTorch](https://pytorch.org/) | Tensors, devices and automatic differentiation of every interface |
| [FINUFFT and cuFINUFFT](https://finufft.readthedocs.io/) | Non-uniform fast Fourier transforms substituted for BART's gridding on the host and on CUDA devices |
| [MRI-NUFFT](https://mind-inria.github.io/mri-nufft/) | Fits the time-segmentation coefficients of {func}`bartorch.linop.FieldCorrected` |
| [TorchSim](https://github.com/FiRMLAB-Pisa/torchsim) | Signal simulation for the signal models of {mod}`bartorch.nlop` |
| [DeepInverse](https://deepinv.github.io/) | Target of {func}`bartorch.interop.to_deepinv`, and a source of denoisers for {class}`~bartorch.priors.ImplicitPrior` |

## Other reconstruction frameworks

| Project | Relationship |
| --- | --- |
| [SigPy](https://sigpy.readthedocs.io/) | Python operators, proximal operators and MRI reconstruction apps, implemented in NumPy and CuPy; an independent implementation for checking conventions |
| [MRpro](https://github.com/PTB-MR/mrpro) | MRI reconstruction implemented in PyTorch itself, with its own operators and data handling |
| [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) | MRI reconstruction in Julia |
| [Pyxu](https://pyxu-org.github.io/) | General computational-imaging framework; its treatment of forward operators, functionals and proximal algorithms is the conceptual reference for {doc}`../explanation/inverse-problems` |
