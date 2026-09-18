# Command line

```{eval-rst}
.. currentmodule:: bartorch.cli
```

`bartorch.cli`. BART's command line, served by this package. Installing
bartorch puts a `bartorch` command on the path, and it takes the arguments
`bart` takes:

```sh
bartorch pics -l1 -r0.01 -i30 kspace sensitivities image
bartorch ecalib -m1 kspace maps
bartorch --list
```

so a script that calls `bart` runs against it with the name changed, or with a
shell alias, and needs no BART installation of its own.

## What runs

Where {mod}`bartorch.apps` has the pipeline, the command line is read into a
Python call and the app runs; everywhere else the command itself runs, in this
process, through the same entry point {mod}`bartorch.tools` uses. The two
answer the same bits -- `tests/test_cli.py` holds the app route against the
command route with `numpy.array_equal` -- so which one ran is a question about
speed and not about the answer, and {func}`route` is what says which it was.

An argument the reader does not express sends the whole command line to BART
rather than being ignored: the command declared it. What the reader knows is
the catalogue's, which is generated from BART's own sources, so there is no
second list of flags here to fall out of date.

An input file that is not there is the one thing named here rather than by
BART, and for a reason: a command that fails while loading its arguments
leaves the library unable to serve the next call in the same process, so a
caller who runs {func}`main` twice would hang rather than see the second
answer. The names are checked against the filesystem first -- a CFL pair, a
RA file, a COO file, or the name itself -- and a missing one is reported
without asking BART.

## Help

`bartorch <command> --help` prints what `bart <command> -h` prints, from the
catalogue rather than from BART, because BART answers its own help by calling
`exit`, which would take the interpreter with it.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   main
   route
   read
```
