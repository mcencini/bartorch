# Command line

`bartorch.cli` implements the `bartorch` console command, which accepts the
command lines of BART's `bart` executable and operates on CFL files:

```sh
bartorch pics -l1 -r0.01 -i30 kspace sensitivities image
bartorch ecalib -m1 kspace maps
bartorch --list
```

A command for which {mod}`bartorch.apps` has a pipeline is parsed into a call
of that app; every other command runs as the BART command, in the same
process, through the entry point {mod}`bartorch.tools` uses.  The options are
read from BART's own command declarations, and an option the app does not
express sends the command line to the BART command.
`bartorch <command> --help` prints BART's help text for the command.

```{eval-rst}
.. currentmodule:: bartorch.cli
```

| Object | Description |
| --- | --- |
| {obj}`~bartorch.cli.main` | Run one command line; returns its exit code |
| {obj}`~bartorch.cli.route` | Whether a command line runs as an app or as the BART command |
| {obj}`~bartorch.cli.read` | Read a CFL pair as a tensor in this package's C-order layout |
