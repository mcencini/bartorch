# Issue reports

Search the [issue tracker](https://github.com/mcencini/bartpy/issues) for the
error and function name before opening a report.  Include:

- expected and observed results, with the complete traceback;
- a minimal script using a synthetic phantom or a random tensor with a fixed
  seed;
- input shapes, dtype, device, axis and trajectory conventions, and
  reconstruction options, including regularization and transform
  normalization;
- OS, Python, PyTorch and bartorch versions, the installation command, and
  `bartorch.build_info()`.  For source builds, the repository and BART commits
  (`git rev-parse HEAD` and `git -C external/bart rev-parse HEAD`);
- for CUDA: GPU model, driver, `torch.version.cuda`, and both availability
  checks.  For non-Cartesian transforms: FINUFFT and cuFINUFFT versions.

If import or loading the library fails, report that error directly;
diagnostics that need the library may fail too.  For numerical discrepancies,
include a reference calculation and the relative error.  For performance
reports, give warm-up, synchronization, thread count, problem size and peak
memory alongside timing.

Share only data you are entitled to publish; a synthetic reproduction is
usually easier to investigate.  Remove patient identifiers, credentials and
private paths from logs.  Report documentation issues with the page URL, the
unclear passage, and the command that failed if there is one.
