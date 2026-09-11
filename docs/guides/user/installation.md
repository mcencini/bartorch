# Installation

## PyTorch first

Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
to install the CPU or CUDA build you want in your active Python environment.
Then install bartorch into that same environment:

```bash
python -m pip install bartorch
```

The metadata requires Python 3.10+, PyTorch 2.1+, NumPy 1.24+ and SciPy 1.10+.
A wheel includes the embedded BART library; wheel users need neither a
separate BART executable nor a C compiler.

:::{note}
This checkout is pre-alpha.  The command above is the intended release install
path, not a claim that wheels for every platform are published.  If pip cannot
find a suitable distribution, use the {doc}`../developer/toolchain` source
installation.  A source archive needs the developer toolchain too.
:::

## Checking the installation

```python
import torch
import bartorch
import bartorch.tools as bt

print(torch.__version__)
print(bartorch.__version__)
print(bartorch.bart_version())
print(bartorch.build_info())
image = bt.phantom([32, 32])
print(image.shape, image.dtype, image.device)
```

## Devices and non-Cartesian transforms

CUDA needs both a CUDA-capable PyTorch and a bartorch library built with CUDA;
`torch.cuda.is_available()` and `bartorch.cuda_available()` check each.  Move
input tensors with `tensor.to("cuda")`.  Some commands stage work through host
memory, so tensor placement alone does not guarantee every step runs on the
card.  CPU and CUDA are the device paths; Apple MPS is not supported.

FINUFFT computes every non-Cartesian transform.  It is a dependency rather
than an extra, so `pip install bartorch` brings it.  cuFINUFFT serves a
transform on a card and is an extra:

```bash
python -m pip install 'bartorch[cufinufft]'
```

Wheels are published for Linux x86_64 and macOS on Apple silicon, the
platforms FINUFFT ships wheels for too.  Elsewhere -- Linux on aarch64, an
Intel Mac -- `pip install bartorch` builds from the source distribution and
builds FINUFFT alongside it, which needs CMake, ninja and a C++ compiler.

The substitution installs itself on first use.  A transform it cannot serve
raises an error naming the reason rather than falling back to BART's own
gridder.  The `mkl` extra is optional; the Cartesian examples do not need it.

## Platforms

Linux is the platform bartorch is developed and measured on, and the one the
CUDA path is written for.

**macOS works, without FINUFFT.**  torch and the FINUFFT wheel each carry an
OpenMP runtime, and LLVM's runtime ends the process rather than run beside a
second copy of itself (`OMP: Error #15`).  On macOS the substitution checks
for that pair before the first call into FINUFFT, declines when it finds it,
and says so once at warning level; BART's own gridder then computes the
non-Cartesian transforms.  The same collision is
[open upstream in mri-nufft](https://github.com/mind-inria/mri-nufft/issues/333)
with no fix.  `KMP_DUPLICATE_LIB_OK=TRUE` makes it run and is documented by the
runtime's authors as unsafe -- a crash later, or a wrong answer quietly -- so
bartorch neither sets nor suggests it.  A conda environment where one OpenMP
runtime serves both packages is the way to FINUFFT on macOS; it is untested
here.

**Windows is not a target.**  BART does not build on it; WSL2 is a Linux
install like any other.

Next: {doc}`conventions` and {doc}`../../auto_examples/index`.
