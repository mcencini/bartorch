# Documentation

Documentation is part of the public interface of a scientific software project. Technical terminology, mathematical definitions, units, shapes, conventions, and behavioral claims should therefore receive the same care as implementation.

Good scientific documentation is not uniformly explanatory. API reference, examples, conceptual explanation, and task-oriented guides serve different purposes. A useful explanation in one context may be distracting or inappropriate in another.

These guidelines define both **what information belongs in each form of documentation** and **how that information should be written**.

The general objective is precise, conventional scientific writing: accessible where explanation is appropriate, concise where reference is required, and technically accurate throughout.

## Documentation structure

Before writing, determine what kind of documentation is needed.

Four broad forms are useful:

| Documentation               | Primary question                                         |
| --------------------------- | -------------------------------------------------------- |
| API reference               | What exactly does this object do?                        |
| Examples and gallery        | What does its use on a representative problem look like? |
| Explanation and concepts    | Why does this concept or method work this way?           |
| Tutorials and how-to guides | How do I accomplish this task?                           |

These forms may link extensively to one another, but they should not be collapsed into a single style of documentation.

### API reference

API reference defines the public interface and its precise semantics.

The intended reader is technically competent and reasonably familiar with the relevant scientific domain. Standard domain concepts should be named directly rather than replaced by informal explanations.

Mature scientific Python projects such as MRpro, PyLops, PyProximal, Pyxu, DeepInv, and SigPy provide useful examples of the desired reference-documentation register. Domain-specific authoritative libraries and specifications should additionally guide terminology and conventions.

A public function, class, method, or operator should document, where relevant:

1. the mathematical, scientific, or computational object represented;
2. the operation performed;
3. parameters and their precise meanings;
4. return values;
5. shapes, dimensions, axes, batching, and broadcasting;
6. physical units;
7. normalization, coordinate, sign, and ordering conventions;
8. a mathematical definition when it improves precision;
9. important algorithmic semantics;
10. restrictions, assumptions, and special cases;
11. differentiation behavior where relevant;
12. correspondence with upstream libraries, specifications, commands, or literature where useful.

Not every item applies to every object. Documentation should communicate the relevant contract rather than mechanically fill a template.

#### Summary lines

A summary line should classify or directly describe an object. It is not a tagline.

Prefer:

> Cartesian SENSE encoding operator.

> Proximal operator of the L1 norm.

> Construct a trapezoidal gradient event.

Avoid:

> Coils, a Fourier transform, and the samples that were taken.

> The penalty that makes things sparse.

> The workhorse gradient builder.

Concise noun phrases are often appropriate for classes and properties. Functions normally use a concise declarative description of their operation.

Established terminology should not be replaced by a description of what the term means merely to make the summary more approachable.

#### Mathematical content

Use rendered equations when they specify behavior more precisely than prose.

For example, if an operator implements

$$
A = PFS,
$$

stating the model and defining \(P\), \(F\), and \(S\) may be clearer than describing the sequence of operations indirectly.

Equations in API reference document semantics. Extended derivations and theoretical motivation generally belong in explanatory documentation.

Normalization, units, signs, coordinate systems, and other conventions must be established before documenting an equation.

#### Implementation details

Implementation details belong in API reference when they affect observable behavior, numerical properties, differentiation, interoperability, performance expectations, or correct usage.

They should not replace the scientific or mathematical definition of the object.

For example, describe an encoding operator mathematically before discussing how its operations are fused or batched internally.

#### Background theory

API reference should not teach standard domain theory from first principles.

When substantial background is useful, provide a concise statement of the relevant semantics and link to explanatory documentation.

### Examples and gallery

Gallery examples show complete, representative scientific uses of the library.

The example galleries of MRpro, MRI-NUFFT, and DeepInv are useful models: executable scientific workflows accompanied by concise methodological explanation.

An example is **not a conversational tutorial**. Code, figures, and scientific results should dominate the page. Prose supplies the context necessary to understand the problem, consequential methodological choices, conventions, and interpretation.

A substantial example will often include:

1. a short statement of the scientific or computational objective;
2. only the background needed for the particular example;
3. data or problem setup;
4. method configuration;
5. execution using the public API;
6. visualization or quantitative evaluation where appropriate;
7. brief interpretation when the result is not self-evident.

These elements need not become separate headings, particularly for short examples.

#### Explain decisions, not execution

Prose should explain decisions whose significance is scientific, mathematical, numerical, or specific to the library.

Useful explanations include:

* why a particular model, operator, solver, acquisition, trajectory, or regularizer is appropriate;
* parameter values that materially affect the result;
* physical or mathematical conventions needed to interpret the example;
* non-obvious preprocessing;
* assumptions and limitations;
* what a figure or quantitative result demonstrates.

Routine Python execution should not be narrated.

Avoid:

> First, we import the required packages.

> Now let's create the operator.

> Next, we run the reconstruction.

> Finally, let's visualize the results.

> This is where things get interesting.

Prefer a concise methodological statement:

> The spiral acquisition is reconstructed with a non-Cartesian SENSE operator. Coil sensitivities are estimated from the calibration data and kept fixed during reconstruction.

Then show the corresponding code.

Imports, straightforward assignments, constructor calls, and plotting commands normally explain themselves.

Coherent code should not be artificially divided into many small blocks merely to create opportunities for explanatory prose.

#### Theory in examples

Include only the theory required to understand the particular example.

An example using a proximal algorithm does not need to introduce proximal optimization from first principles. An example specifically investigating or comparing proximal algorithms may require more theoretical context.

Broader derivations, terminology, and conceptual comparisons belong in explanatory documentation.

Similarly, examples should not reproduce exhaustive API semantics. Link to the API reference instead.

#### Code comments

Comments should explain non-obvious scientific or numerical choices rather than translate code into English.

Avoid:

```python
# Create the trajectory.
trajectory = make_trajectory(...)
```

Prefer, when the information matters:

```python
# Use 24 interleaves to obtain the target sampling density.
trajectory = make_trajectory(...)
```

Comments that merely restate the following expression should normally be omitted.

#### Results

Figures and quantitative outputs should answer a scientific or computational question rather than merely demonstrate that plotting is possible.

Provide enough axis labels, units, legends, captions, or surrounding context to make results interpretable.

Interpret results conservatively. A single example should not be generalized into a broad performance claim.

Avoid:

> As we can clearly see, the reconstruction is much better.

Prefer a concrete observation or quantitative measure when interpretation is needed.

#### Tone

Gallery prose can provide more context than API reference, but should remain concise, technical, and restrained.

Avoid:

* second-person narration unless genuinely useful;
* "we" merely to narrate execution;
* rhetorical questions;
* enthusiasm and promotional language;
* conversational transitions;
* motivational filler;
* metaphors and personification;
* repeated previews and recaps;
* prose whose only purpose is to connect adjacent code blocks.

The intended result is **a reproducible scientific example with concise methodological annotation**, not a lesson delivered by a narrator.

When a sentence merely describes what the next line of code does, it can usually be removed.

### Explanation and concepts

Explanatory documentation answers questions about concepts, theory, terminology, relationships between methods, architecture, and design rationale.

Pyxu's conceptual introductions to inverse problems and proximal optimization are a useful model for the desired depth and register.

This is the appropriate place for material that would be too pedagogical for API reference or too general for an example.

Explanatory documentation may:

* introduce terminology to readers outside the immediate specialty;
* establish mathematical notation;
* derive or motivate formulations;
* explain relationships between concepts;
* compare related approaches;
* connect library abstractions to underlying theory;
* explain conventions and their consequences;
* discuss architectural or API design decisions;
* explain numerical or scientific trade-offs;
* connect implementations to literature or upstream software.

Where possible, begin from the scientific, mathematical, or computational concept rather than from the Python class hierarchy.

For example, an explanation of an inverse-problem library might first introduce

$$
y = A(x) + \varepsilon
$$

and a regularized formulation such as

$$
\min_x D(Ax, y) + \lambda R(x),
$$

before explaining how the library represents \(A\), \(D\), and \(R\).

#### Accessibility and terminology

Explanatory documentation may assume scientific and mathematical literacy without assuming expertise in the immediate subfield.

For example, an MRI researcher may need proximal optimization explained, while a numerical optimization researcher may need SENSE explained.

Accessibility should come from **introducing and explaining the standard terminology**, not from avoiding it.

Prefer:

> The gradient moment is the time integral of the gradient waveform. Its zeroth moment determines the corresponding displacement in k-space.

Avoid:

> The gradient's moment is essentially how much gradient has accumulated.

Once a term has been introduced, use it consistently. Do not repeatedly substitute descriptive phrases for it.

Explanatory prose may be more discursive than API reference, but it should remain scientific prose. Explanation is not a license for conversational, literary, or promotional writing.

#### Depth

Explain enough to make the concept understandable and to establish why it matters to the software.

Avoid expanding into textbook material unrelated to the project.

Equations, diagrams, figures, references, and worked reasoning are encouraged when they materially improve understanding.

### Tutorials and how-to guides

Tutorials and how-to guides provide procedural guidance.

A tutorial teaches a workflow, often from a relatively clean starting point. A how-to guide answers a focused practical question.

Examples include:

> How do I define a custom operator?

> How do I reconstruct non-Cartesian multi-coil data?

> How do I export a sequence?

State prerequisites, provide a reliable procedure, and explain non-obvious choices.

Do not reproduce detailed theoretical material or exhaustive API documentation. Link to explanation and reference pages where appropriate.

As with gallery examples, avoid narrating obvious operations merely to create prose between steps.

## Separation of concerns

Useful information should live where readers are most likely to need it.

Documentation types should therefore link to one another rather than duplicate one another.

* API reference links to explanation for theory and terminology.
* API reference links to examples for complete workflows.
* Examples link to reference for exhaustive interface semantics.
* Examples link to explanation for broader theoretical context.
* Explanation links to reference for concrete software interfaces.
* Tutorials and how-to guides link to explanation rather than reproducing derivations.

A paragraph that answers the wrong question for its current page should usually be moved or replaced by a link rather than forced into the local prose style.

In particular, useful explanatory material discovered while simplifying an overloaded docstring need not be discarded. It may form the basis of an explanatory page.

## Scientific writing

### Terminology

Use terminology established by the relevant scientific, mathematical, numerical, and software domains.

Do not invent synonyms merely to improve prose rhythm.

Prefer established terms such as:

> auxiliary variable
> adjoint operator
> gradient moment
> sampling trajectory
> proximal operator

over paraphrases such as:

> an extra unknown
> the operator going backwards
> what the gradient accumulates
> where the samples go
> what the solver applies to the penalty

A technical noun should not normally be replaced by a relative clause describing what the object does.

If a term may be unfamiliar to some readers, define it in explanatory documentation. Do not avoid the term in reference documentation.

Project-specific terminology and conventions should be documented explicitly and used consistently.

### Prose

Prefer precise, declarative scientific prose.

Complete sentences are generally preferable to rhetorical fragments. Concision is desirable when it does not remove relevant information.

Avoid:

* metaphors and personification for technical objects;
* taglines;
* promotional or journalistic prose;
* conversational narration;
* invented informal terminology;
* stylistic synonyms for established technical terms;
* vague words such as "thing", "stuff", or "piece";
* constructions such as "X is what Y does/takes";
* "Reach for X...";
* "Think of X as..." when a direct definition is available;
* filler such as "simply", "just", "basically", and "under the hood".

For example:

> A tensor captured by the callback's closure is not an operator input, so no cotangent is propagated to it.

is preferable to:

> A weight the prior closes over reaches no gradient.

Likewise:

> Gradient waveform continuity across block boundaries.

is preferable to:

> Whether each gradient carries on from the block before it.

Technical writing need not be deliberately opaque or terse. The objective is clarity through correct terminology and explicit relationships rather than through conversational paraphrase.

### Units, shapes, and conventions

Units, shapes, axes, normalization, coordinate systems, sign conventions, ordering, and discretization rules are part of the interface whenever they affect interpretation or behavior.

Document them explicitly where relevant.

Do not infer units or conventions from parameter names alone.

When internal and external representations use different conventions, state the distinction.

Use one notation consistently for the same physical quantity unless there is a technical reason not to.

### Mathematical notation

Use conventional mathematical notation.

Rendered mathematics is preferred when it makes a definition or relationship substantially clearer than prose or a monospace pseudo-equation.

In explanatory material:

* introduce symbols before relying on them;
* define important quantities;
* state relevant assumptions;
* provide enough surrounding explanation to interpret the equations.

In API reference, equations primarily specify semantics and normally should not become extended derivations.

Mathematics should not be added merely to make documentation appear more rigorous.

### Scientific claims and references

Distinguish among:

* behavior established by the implementation;
* mathematical properties;
* empirical observations;
* literature-derived claims;
* assumptions;
* recommendations.

Cite primary literature or authoritative specifications for nontrivial external scientific claims where practical.

Do not turn results from one example, benchmark, dataset, or configuration into unsupported general claims.

Do not describe an estimate as a measurement, or a limited check as a general guarantee.

## Docstrings

Docstrings are primarily API reference.

They should make the public contract clear without attempting to contain every useful piece of background information.

Use the project's established docstring convention consistently.

### Parameters and returns

Parameter documentation should state semantic meaning rather than merely repeat the type or parameter name.

Avoid:

> `axis`
> The axis.

Prefer:

> `axis`
> Spatial axis along which the finite difference is evaluated.

Document units, allowed values, shapes, broadcasting behavior, or conventions when these are not obvious from the type.

Return documentation should similarly describe what is returned rather than repeat the return type.

### Notes

Use Notes for information that materially affects interpretation or use of the API but does not belong naturally in Parameters or Returns.

Examples include:

* mathematical definitions;
* normalization conventions;
* algorithmic restrictions;
* correspondence with upstream implementations;
* numerical considerations;
* differentiation behavior.

Do not use Notes as a place for general background theory that belongs in explanatory documentation.

### Docstring examples

Docstring examples should be short and API-focused.

Useful examples demonstrate:

* basic invocation;
* important shapes;
* return structure;
* conventions that are easier to show than describe;
* a common non-obvious case.

Long scientific workflows belong in the example gallery.

Conceptual derivations belong in explanatory documentation.

Prefer executable examples where practical.

An example should not be added solely to satisfy a template when it provides no useful information beyond the signature.

## Developer documentation

Developer documentation may discuss implementation architecture and internal abstractions more directly than user-facing documentation.

Use the same precise terminology and restrained prose.

Explain rationale where it helps maintainers understand why an architecture, representation, or constraint exists.

Distinguish current design rationale from historical narration.

Abandoned alternatives, transient development reasoning, and change history generally belong in issue trackers, design records, release notes, or repository history unless they are necessary to understand a current constraint.

## Editing existing documentation

Documentation maintenance is semantic editing, not indiscriminate rewriting.

Before making a substantive change:

1. identify the type and purpose of the documentation;
2. establish the behavior being documented;
3. consult implementation, tests, specifications, upstream documentation, or literature where necessary;
4. preserve technically useful existing material;
5. correct terminology and conceptual errors;
6. remove inappropriate prose rather than replacing it with stylistic synonyms;
7. avoid unrelated stylistic churn.

Existing prose should not be assumed to be correct merely because it is already published.

Conversely, documentation that is already precise and appropriate should not be rewritten merely to impose a different wording.

When documentation and implementation appear to disagree, establish the intended behavior before changing the documentation.

## Building and reviewing documentation

Substantial documentation changes should be reviewed in rendered form.

Where supported by the project:

1. build the complete documentation;
2. run doctests and executable examples;
3. run documentation-specific tests and checks;
4. inspect rendered pages;
5. check equations, figures, tables, links, and cross-references;
6. review terminology for consistency.

A successful documentation build establishes that the documentation can be rendered; it does not establish that the documentation is scientifically or editorially correct.

## Review checklist

### API reference

* Is the established technical term used?
* Is the object or operation defined directly?
* Are units, shapes, conventions, and restrictions unambiguous?
* Is mathematical notation used where it improves precision?
* Is background theory limited to what is needed to specify the interface?
* Have implementation details displaced the scientific abstraction?

### Examples and gallery

* Is this a meaningful and reproducible scientific workflow?
* Do code and results dominate the page?
* Does prose explain consequential choices rather than narrate execution?
* Is theoretical background limited to what the example needs?
* Are results interpreted concretely and conservatively?
* Does the example demonstrate recommended public-API usage?

### Explanation

* Does the page teach the underlying concept rather than narrate the API?
* Are standard terms introduced accurately and then used consistently?
* Are motivation, relationships, assumptions, and design choices clear?
* Is the material accessible without sacrificing technical terminology?
* Does the depth remain relevant to the software?

### Tutorials and how-to guides

* Is the task or workflow clear?
* Are prerequisites stated where necessary?
* Does the guide explain non-obvious choices without narrating obvious code?
* Does it link to reference and explanation instead of duplicating them?

### All documentation

* Is the content technically correct?
* Is terminology conventional and consistent?
* Are important units and conventions explicit?
* Is any wording present primarily to sound clever, friendly, vivid, or varied?
* Has precision been sacrificed for accessibility?
* Is the material in the appropriate documentation type?
* Would removing a sentence lose useful scientific, mathematical, API, or procedural information?

When accessibility and terminology appear to conflict, **retain the standard terminology and explain it better**.
