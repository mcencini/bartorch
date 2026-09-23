# Contributors and citation

bartorch is developed by its
[contributors](https://github.com/mcencini/bartorch/graphs/contributors)
(`git shortlog -sne HEAD` in a clone) and builds on the work of the BART
developers, whose history is that of the `external/bart` submodule.
{doc}`../guides/developer/index` describes how to contribute.

## Citing bartorch

bartorch has no publication and no archival DOI.  Work that uses it cites BART,
the backends that computed its results, and the methods it applied, and
reports the software versions:

- **BART**: Uecker M, Ong F, Tamir JI, Bahri D, Virtue P, Cheng JY, Zhang T,
  Lustig M. Berkeley Advanced Reconstruction Toolbox. *Proc Intl Soc Mag Reson
  Med* 23:2486 (2015); and the Zenodo record of the BART version used, reached
  from [doi:10.5281/zenodo.592960](https://doi.org/10.5281/zenodo.592960).
  BART asks that the articles corresponding to the methods used be cited as
  well; its `doc/references.txt` lists them.
- **Methods**: the publications of the reconstruction methods used, for example
  SENSE, ESPIRiT, compressed sensing or nonlinear inversion; the references at
  the end of each {doc}`explanation page <../explanation/index>` and example
  give them.
- **Non-uniform FFT**, for non-Cartesian data: Barnett AH, Magland J,
  af Klinteberg L. A parallel nonuniform fast Fourier transform library based
  on an "exponential of semicircle" kernel. *SIAM J Sci Comput*
  41(5):C479-C504 (2019), [doi:10.1137/18M120885X](https://doi.org/10.1137/18M120885X);
  and, for CUDA tensors, Shih Y, Wright G, Andén J, Blaschke J, Barnett AH.
  cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs.
  *IEEE IPDPSW* 688-697 (2021),
  [doi:10.1109/IPDPSW52791.2021.00105](https://doi.org/10.1109/IPDPSW52791.2021.00105).

## Reproducibility

A reproducible report states:

| Item | Source |
| --- | --- |
| bartorch version or commit | `bartorch.__version__`, `git rev-parse HEAD` |
| BART revision | {func}`bartorch.bart_version`; `git -C external/bart rev-parse HEAD` in a source build |
| Build and backends | {func}`bartorch.build_info`, {func}`bartorch.backend_sources`, the FINUFFT and cuFINUFFT versions |
| Device | CPU or GPU model, driver and CUDA versions |
| Data conventions | Array order, trajectory units, Fourier-transform centring and normalization |
| Calibration and sampling | Sensitivity estimation, sampling pattern or trajectory |
| Solver settings | Algorithm, iterations, step size, regularization terms and weights, data scaling |
