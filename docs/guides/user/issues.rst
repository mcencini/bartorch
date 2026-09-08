Reporting issues
================

Search the `issue tracker <https://github.com/mcencini/bartpy/issues>`_ for the
error and tool name before opening a report. Include:

* Expected and observed results, including the complete traceback.
* A minimal script using a synthetic phantom or random tensor with a fixed seed.
* Input shapes, dtype, device, axis/trajectory conventions and reconstruction
  options, including regularization and transform normalization.
* OS, Python, PyTorch and bartorch versions, installation command, and
  ``bartorch.build_info()``. For source builds include the repository and BART
  commits (``git rev-parse HEAD`` and ``git -C bart rev-parse HEAD``).
* For CUDA: GPU model, driver, ``torch.version.cuda``, and both CUDA availability
  checks. For NUFFT: FINUFFT/cuFINUFFT versions and configuration.

If import or library loading fails, report that error directly; diagnostics
requiring the library may fail too. For numerical discrepancies, include a
reference calculation and relative error. For performance reports, give warm-up,
synchronization, thread count, problem size and peak memory alongside timing.

Share only data you are entitled to publish; synthetic reproductions usually
make issues easier to investigate. Remove patient identifiers, credentials and
private paths from logs. Report documentation issues with the page URL, unclear
passage, and command that failed if applicable.
