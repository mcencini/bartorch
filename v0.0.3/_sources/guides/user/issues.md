# Reporting issues

Bugs, documentation defects and feature requests are filed on the
[issue tracker](https://github.com/mcencini/bartorch/issues/new/choose), whose
forms ask for the information below.  Search the existing issues for the error
message and the function name first.

A bug report contains:

- the observed and the expected behaviour, with the complete traceback;
- a minimal script on synthetic data — {func}`bartorch.tools.phantom`, or a
  random tensor with a fixed seed — rather than acquired data;
- the shapes, dtypes and devices of the inputs, the axis and trajectory
  conventions assumed, and the options used, including regularization terms
  and Fourier-transform normalization;
- the bartorch version (`bartorch.__version__`) or commit, the output of
  {func}`bartorch.build_info` and {func}`bartorch.bart_version`, the Python and
  PyTorch versions, the operating system and the installation command;
- for a source build, the repository and BART commits:
  `git rev-parse HEAD` and `git -C external/bart rev-parse HEAD`;
- on a CUDA device, the GPU model, the driver version, `torch.version.cuda`,
  and the values of `torch.cuda.is_available()` and
  {func}`bartorch.cuda_available`;
- for non-Cartesian problems, the FINUFFT and cuFINUFFT versions.

If importing bartorch or loading its library fails, report that error; the
diagnostic functions above need the library.  A numerical discrepancy report
states the reference it was compared with — an independent implementation, an
explicit sum, a closed form — and the relative error observed.  A performance
report states the problem size, the thread count, whether the timing includes
warm-up and device synchronization, and the peak memory.

Share only data you may publish.  Remove patient identifiers, credentials and
private paths from logs.  Report a security vulnerability privately, as
described in {doc}`security`, not as an issue.
