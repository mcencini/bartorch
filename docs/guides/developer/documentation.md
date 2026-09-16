# Documentation

## Building

From the repository root, in a Python environment with PyTorch (a CPU build is
enough):

```bash
python -m pip install -r docs/requirements.txt
./scripts/build_docs.sh
```

Open `docs/_build/html/index.html`.  The build imports `bartorch` from `src/`,
which needs torch but not the compiled library.  Warnings are errors, as in the
documentation workflow.  `--online` fetches intersphinx inventories.  Check
external links with `python -m sphinx -b linkcheck docs docs/_build/linkcheck`.

## Pages and the reference

Pages are Markdown, parsed by MyST; a directive is written as a fenced block,
and reStructuredText only inside `{eval-rst}`.  Each page of the API reference
(`docs/api/`) is a set of sections, each an `autosummary` table of the names it
documents; autosummary writes a page per name into `docs/api/generated/`, and
autodoc renders its docstring.  `tests/test_docs.py` fails when a public name is
missing from the reference or the reference lists one that is not public.

A command without a hand-written wrapper is built at import from BART's
catalogue, with BART's help as its docstring; its reference page is rendered
from that docstring like any other.

## Audience and register

The reader is a researcher or engineer who already knows MRI reconstruction and
numerical optimization.  Write for that reader.  Do not explain Fourier
transforms, proximal operators or least squares from first principles; explain
what this library does and what conventions it fixes.

The register to aim for is that of BART, MRpro, SigPy, DeepInv, PyLops,
PyProximal and Pyxu: conventional terminology, explicit mathematics, precise API
semantics, minimal ornament.  Dry and unremarkable is the target.  If a sentence
reads as conspicuously written, rewrite it until it does not.

Use the established term for the established concept, and repeat it:

> forward and adjoint operators; encoding operators; Cartesian and
> non-Cartesian sampling; sampling operators and trajectories; SENSE and coil
> sensitivity maps; FFT and NUFFT; proximal operators; regularization
> functionals; auxiliary variables and variable splitting; primal and dual
> variables; data-fidelity terms; linear least squares and regularized inverse
> problems; temporal and subspace bases and coefficient representations;
> iterative algorithms; fixed-point and implicit differentiation.

Repeating a technical term is correct.  Varying it for rhythm is not: a
synonym invented to avoid repetition reads as a second concept.

Prohibited, as a class and not only in these exact forms:

- sentence fragments used as summaries or headings;
- metaphor, personification and analogy ("the operator owns its sampling", "the
  term carries no derivative", "another term is furniture");
- the constructions `X is what Y does`, `X is what Y takes`, `which is what Y
  runs`, `Reach for X when ...`, `X, Y, and the Z that ...`;
- pseudo-spoken English, journalistic transitions, conversational imperatives;
- "simply", "just", "basically", "under the hood", "quietly";
- invented informal terminology where a conventional category exists;
- vague `thing`, `stuff`, `it` or `that` where a technical noun is available;
- prose written for rhythm, voice or memorability.

## What a reference entry establishes

For a public function, class or operator, document what applies of:

1. the mathematical or computational object it represents;
2. the operation or model it implements;
3. inputs and outputs;
4. shapes and dimensions;
5. normalization and conventions;
6. batching behaviour;
7. restrictions and special cases;
8. differentiation and autograd behaviour;
9. the corresponding BART functionality.

A summary line is a classification, not a tagline.  Write
`Cartesian SENSE encoding operator.`, not
`Coils, a Fourier transform, and the samples that were taken.`

Use mathematics where it is more precise than prose -- an encoding operator's
forward model, a solver's objective, a regularization functional.  `sphinx.ext.mathjax`
is enabled, so `.. math::` renders in the reference.  Use it where an equation
defines the object; leave incidental expressions in monospace.  Do not convert
every expression mechanically, and do not derive: a reference entry states the
formulation, it does not prove it.

A reference entry is not a tutorial.  Examples are short and show a call, not a
workflow.

## Differentiation

These are different claims and must not be conflated:

- the operation is not differentiable;
- no backward pass is implemented for it;
- a value is deliberately detached;
- an object is held fixed while something else is differentiated;
- the gradient comes from unrolling the iteration;
- the gradient comes from implicit differentiation at a fixed point.

Name which one applies.  Do not write that something "carries no derivative" or
that "the gradient is meant to reach" somewhere.

## BART correspondence

Keep three things apart:

1. the mathematical operation;
2. this library's Python abstraction;
3. the BART command, option or internal primitive behind it.

State the correspondence; do not let it stand in for the definition.  A BART CLI
option is not a mathematical description.  Write

> With a temporal basis the optimization variable holds subspace coefficients,
> which the basis operator maps to the acquired frames before the sampling
> operator is applied, as in `bart pics -B`.

and not `what pics -B takes`.

Naming the BART function, file or line behind a restriction is valuable and
should be kept -- `iter2_ist` ignoring a term's transform, `linop_stack_cod`
making the normal of a stack the sum of the parts'.  That is correspondence, not
paraphrase.

## Source of truth

Existing prose in this repository is not a style reference and is not a reliable
description of semantics.  Before rewriting a description, read the
implementation, read the tests, and read upstream BART where the behaviour is
BART's.  Then write what it does.

Do not invent detail to round a description out.  Where the semantics stay
unclear, say so in the pull request rather than guessing.

## Docstrings

Docstrings are NumPy style.

- State what is not obvious from the name, the signature and the annotations:
  units, coordinate frames, composition order, invariants, side effects,
  preconditions, special return values, and behaviour a maintainer could break
  by accident.
- Do not restate the signature, narrate the implementation, or document a
  parameter with its own name.  List a parameter only when its entry adds
  something.
- A module docstring states the module's responsibility in a sentence or a few.
  Design rationale belongs in `AGENTS.md` or in a comment beside the code it
  explains.
- A private helper gets a docstring only for a contract that is not obvious.
- Precision over brevity: a short paragraph for a real invariant is better than
  an ambiguous line.

A class docstring says what abstraction the class represents and which
conventions it fixes.  Document constructor parameters and public attributes
where their meaning is not obvious, and for a stateful class the state whose
interpretation or lifecycle would otherwise be unclear; do not enumerate every
attribute mechanically.  A large `Parameters` or `Attributes` section that
mostly restates type annotations is worse than no section.

A function docstring says what the operation means, not how it is carried out.

Docstrings carry the contract; comments carry the implementation.  Algorithmic
tricks, performance-sensitive choices and the reason a piece of code is written
in a non-obvious way belong in a comment beside it, and do not get moved into a
docstring to preserve them.

## Editing existing documentation

Clean up documentation next to code you are changing, but do not widen a focused
change into a repository-wide rewrite unless that is the task.  Leave correct,
conventional prose alone: shape tables, parameter lists and BART correspondence
notes that already read well are not improved by restatement.
