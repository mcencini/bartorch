# License and third-party notices

bartorch is distributed under the MIT license below.  The components it
embeds or vendors keep their own licenses, which the MIT license does not
replace; their notices are distributed with the source and, from this
version on, in each wheel's `.dist-info/licenses/` directory.

```{literalinclude} ../../LICENSE
:language: text
```

## Components compiled into the library

| Component | Location | License |
| --- | --- | --- |
| BART | `external/bart/` (Git submodule) | BSD-3-Clause, `external/bart/LICENSE`, with further notices in individual files |
| pocketfft | `external/pocketfft/` | BSD-3-Clause, `external/pocketfft/LICENSE.md` |
| BlocksRuntime (LLVM compiler-rt) | `external/blocksruntime/` | University of Illinois/NCSA or MIT, at the user's choice, `external/blocksruntime/LICENSE.TXT`; linked into Linux builds made with clang |
| GNU OpenMP runtime (`libgomp`) | Added to the Linux wheel by `auditwheel` | GPL-3.0 with the GCC Runtime Library Exception |

BART's license:

```{literalinclude} ../../external/bart/LICENSE
:language: text
```

## Dependencies

The following are installed as separate packages, each under its own license,
and are not redistributed by bartorch: PyTorch, NumPy, SciPy, FINUFFT and
cuFINUFFT (Apache-2.0), TorchSim, MRI-NUFFT, and the optional MKL and
DeepInverse.  The CUDA wheel links the CUDA runtime, cuFFT and cuBLAS
dynamically and does not contain them.

## Logo

The bartorch logo and mark combine three elements, which
`scripts/make_artwork.py` draws:

| Element | Source | Terms |
| --- | --- | --- |
| The BART mark: the letters and the corner brackets | The outline in BART's `src/geom/logo.c` | BSD-3-Clause, BART's license above |
| The flame | The PyTorch logo, `assets/images/logo-icon.svg` of [pytorch/pytorch.github.io](https://github.com/pytorch/pytorch.github.io), reproduced unaltered in shape and colour and only scaled | A trademark of The Linux Foundation |
| The word `torch` | Glyph outlines of DejaVu Sans Bold | Bitstream Vera license; DejaVu's changes are in the public domain |

PyTorch, the PyTorch logo and any related marks are trademarks of The Linux
Foundation.  bartorch is an independent project and is not affiliated with or
endorsed by the BART developers, the PyTorch Foundation or The Linux
Foundation.
