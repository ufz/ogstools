# Shared rules

These rules apply to OGSTools, ogsworkflows and the other Python projects around OpenGeoSys (OGS).
They follow the conventions of OGS and use OGSTools as the shared tool for running OGS, reading meshes, units and plots.
[OGSTools](ogstools.md) adds the rules that apply only to it.

## Simple code

- Write flat code: early `return`/`continue`, positive conditions, `elif` or `match` instead of nested `if`/`else`, a dict instead of an `if` chain over keys.
- Build a list, dict or set with a comprehension.
  Use a plain loop when the loop body changes something else, such as an existing object or a file.
- Use vectorised numpy and the standard library (`np.asarray`, `np.clip`, `np.isclose`, `itertools.pairwise`) instead of Python loops over arrays.
- Check `is None` explicitly when `0` or `""` are valid values.
- Do not pass arguments that equal the default, unless it makes the call clearer.
- Give non-obvious numbers a name (`rel_tol = 1e-6`). Say in a comment why the value was chosen.
- Keep functions and modules small.
  Put helpers at module level as private functions, not as nested `def`s.
- Use what already exists before you write new code.
  First look in the package itself, then in the installed libraries, then in the standard library.
- Add a shared helper only when a second caller exists.
- A bug fix fixes the cause in the shared code, not the symptom at one caller.
  Check every caller of the function you change.
- Run OGS and its tools through OGSTools (`ot.Model`, `ot.cli()`), not through `subprocess`.
- Do not convert units by hand.
  Use `Variable` (`data_unit`, `output_unit`) or the unit registry `ogstools.variables.u_reg`.

## Names

- Use descriptive names: no single letters except loop indices and common math symbols (`x`, `t`), no unknown abbreviations, no `_temp` or `_tmp` suffixes for intermediate values, plurals for collections, booleans that read as predicates (`is_`, `has_`, `show_`).
- Name functions after what they do:
  - A free function has no `get_` prefix.
  - `compute_` and `to_` return a new object.
  - `add_`, `set_` and `update_` are methods that change `self`.
  - `read` and `save` mean file input and output.
- Name alternative constructors `from_<source>` (`from_file`, `from_data`).
- Use `snake_case` for modules, functions and variables, also for acronyms (`read_csv`, `to_vtk`).
  Use `CapWords` for classes and `UPPER_CASE` for module-level constants (`DATA_DIR`).
- Use one term and one spelling per concept in code, docs, examples and data files.
- Do not repeat the module name in a function name.

## Public API

- The public API is what the package and each subpackage `__init__.py` export in `__all__`.
  Everything else is internal, whatever its module is called.
- Give new modules and helpers a leading `_`.
  Export the user-facing names from the subpackage `__init__.py` with a relative import, and list them in `__all__`.
- Make something public only when users need it. Public means documented, tested and stable.
- A public parameter accepts each form in which users have the value at hand.
  The function converts it once at the start, and internal helpers take only the converted type.
  Common forms:
  - the object or its name or path as `str` (`Path | str`);
  - the object or the file it is read from;
  - one item or a sequence of items;
  - a list or an array, a plain number or a quantity with a unit.
- Type parameters that are only iterated or indexed as `Sequence[...]`, not `list[...]`.
- Return the natural, specific object: a `Path` or `list[Path]` of written files, a `Figure`, the domain object.
  Do not return an opaque tuple or a broad union.
- Keep signatures and return values consistent with sibling functions.
  Follow Python conventions: `append` and `extend` change the object, `__add__` returns a new one.
- Offer alternative constructors as classmethods of the core class.
- Do not change arguments. Work on a copy and return it.
  Only methods change an object, only `self`, and the name and docstring say so.
- Make copy semantics explicit: `copy(deep=True)`, or `copy(deep=False)` with a comment why a shallow copy is enough.
- Prefer explicit over magic:
  - separate public functions instead of a dispatch on the input shape,
  - named options (`x_metric="model_time"`) instead of boolean switches,
  - a required argument where a wrong default would silently give a wrong result.
- Use explicit, typed parameters for the options users normally set.
  Use `**kwargs` only for option sets too large to list, such as plot styling. Document the accepted keys.
- Use `None` as the default for values that are resolved at call time. Never put a configuration value into a default.
- Prefer the simplest interface that covers today's use cases (`str` before `tuple[str, ...]`). Extend it when the need arises.
- Add no option, parameter or abstraction that no current use case needs.
- Model structured data with enums, dataclasses or `NamedTuple`, not with magic strings, positional lists or nested dicts.
  Make a dataclass `frozen=True` unless its instances must change.
- Expose derived values as read-only `@property`s, without a setter.
- Library functions compute and return objects.
  Writing files, printing and parsing arguments belong to a `save` method or a command-line tool.
- Do not read mutable module-level objects in a function. Pass them in.
- Importing the package changes no global state, for example no `setLevel` on a logger.
- Hide file-format details. User code is the same for PVD and XDMF.
- Read and change plot styling only through `ot.plot.setup`, at call time.
  Label axes with `Variable.get_label` (`name $symbol$ / unit`), as `ot.plot` does.

## Changing existing API

- Do not break interfaces or move modules without need.
- Code that you move or merge keeps its input checks and error handling.

## Typing

- Make hints precise, but not narrower than the accepted input: `Literal` for options, `Sequence`, `Callable`, `-> None`.
- Annotate arrays as `np.ndarray` and `**kwargs` as `**kwargs: Any`.
- Annotate alternative constructors `-> Self`.
- Write `Path | str` in this order.
- A module that needs forward references starts with `from __future__ import annotations`. Do not quote annotations.
- Give each `# type: ignore` a comment that says why. Fix the typing where possible.
- Annotate meshes as `pv.UnstructuredGrid`.
  Use `pv.DataSet` only where any dataset type is accepted.

## Errors, warnings and logging

- Check user input at the entry point, before expensive work.
- Handle every expected case explicitly. A final `else` raises.
- Raise built-in exceptions: `ValueError`, `TypeError`, `KeyError`, `FileNotFoundError`.
- Define an exception class only when callers must catch that error apart from the others.
  It derives from the fitting built-in exception, so that catching the built-in still works.
- Never raise `AssertionError`, and never raise a warning class as an exception (`raise UserWarning(...)`).
- The message names the wrong value and the remedy, for example which function lists the valid names.
- Report something the caller should change with `warnings.warn`.
  Report something the caller cannot change with `logger.warning`.
- Log through a module-level `logger = logging.getLogger(__name__)`.

## Comments

- Comment why, not how: intent, workarounds, sign and index conventions, origin of data, sources of formulas and coefficients.
  Give the why in one clause. Longer reasoning goes into the developer guide, and the comment links it.
- Delete comments that repeat the code.
- A comment states how things are, not when they were measured.
  If the reader cannot tell that it describes the current state, it names the commit or MR with the measurement, not a date.
- A `TODO` says what is to do. Link the issue if there is one (`# TODO: …, see #165`).
  A `TODO` never appears in a rendered docstring.
- Before you ask for review, remove unused functions and files.
- Code that exists only to support an older OGS version gets `# TODO(ogs-version): <when it can go>`.
  At each release, maintainers search for this tag and remove the code that the minimum OGS version no longer needs.

## Imports and dependencies

- Use absolute imports (`from <package>.<subpackage>.x import y`). Only `__init__.py` files use relative imports.
- Keep importing the package fast.
  Import the standard library and numpy at module top.
  Import heavier packages inside the function that needs them.
  Put imports that are needed only for type hints under `if TYPE_CHECKING:`.
- Declare every directly imported package in `[project.dependencies]` or in an optional extra, with a lower bound (`numpy>=1.24`).
  Add an upper bound only with a linked issue.
- Do not declare test or docs tooling as a runtime dependency. The `test`, `docs` and `dev` extras need no bounds.
- Do not add a dependency for what a few lines of code can do.
- Put a non-core dependency into an optional extra.
  On `ImportError`, raise with the install hint (`pip install <package>[<extra>]`).
- Do not copy external code into the package and do not add submodules.
  Fix problems upstream. Keep a local workaround only while released versions need it.
- Add no license files besides the SPDX header.

## Tests

### What to test

- Every new or changed code path has a test.
  Documentation examples do not count. An untested feature is not published.
- Cover all supported input variants, for example every supported file format.
- Test at least one failure case: invalid input raises a meaningful error.
  Where it matters, also test boundary values and that inputs stay unchanged.

### How to test

- Put tests in a flat `tests/test_<subpackage>[_<topic>].py`.
  Use a subdirectory only for a subpackage with several modules of its own.
- Prefer plain pytest functions to test classes for new tests.
- Define a fixture in the module that uses it.
  Move it to `tests/conftest.py` as soon as a second module needs it.
- Use `pytest.mark.parametrize` instead of copied tests.
  Give `ids=` when a value is not a str, number or dict, because the id becomes part of baseline file names.
- Extend the existing test of a function before you add a near-identical one.
- Compare arrays with `np.testing.assert_allclose(actual, expected, rtol=..., atol=...)` and scalars with `pytest.approx(expected, rel=...)`.
  Always give an explicit tolerance. Use `np.testing.assert_array_equal` for exact integer data.
- Assert concrete outcomes: values, properties or files that must exist.
  Never wrap a test in `try`/`except`.
- Write output to `tmp_path`. Reset global state with `monkeypatch`, so tests can run in parallel with `pytest-xdist`.
- Give every `skipif` and `xfail` a `reason=` that names the missing tool or the known defect.
- Keep test data small: coarse meshes, at most about ten timesteps.
  Prefer data generated by a script over committed binary files.
- Keep a test without marker within 10 s. pytest lists the tests that take longer at the end of a run.
  Make a slower test smaller: a coarser mesh, fewer timesteps, fewer parameter cases.
- Pass non-obvious values as keyword arguments.
- Mark a test that runs a simulation or an external tool with the marker of the project.

### Test docstrings

A test is named after the function it tests.
Its docstring says in one sentence what the test is for.
A field line with `scope` and `oracle` is recommended for new tests:

```python
def test_read_xdmf():
    """Reading an XDMF file gives the same mesh as reading the VTU file.

    scope: component (read) | oracle: consistency, VTU result
    """
```

- The sentence does not repeat the name or the code.
- `scope` is the [ISTQB test level](https://astqb.org/2-2-test-levels-and-test-types/) and the object under test:
  - `component`: one function, method or class,
  - `component integration`: several components together,
  - `system`: the whole program or workflow, such as a simulation run.
- The level follows the object under test, not the cost of the test.
  A test of a plot function stays `component`, also when a simulation produces its input.
- `oracle` is what decides pass or fail:
  - `reproducibility`: a committed reference or baseline,
  - `correctness`: a closed-form or simplified solution,
  - `consistency`: the result of another path to the same answer, such as another file format or another solver setting,
  - `plausibility`: bounds that follow from numerics or physics,
  - `implicit`: nothing; the test only checks that nothing is raised.
- An optional third field `input` says what the test costs when the level does not show it, for example `input: simulation run` for a component test.
- Put no tolerances or numbers in the docstring. They live in the code.
- Each test docstring stands alone. A sentence that adding or removing another test would make false belongs in the module docstring.
- If a test module needs an overview, its module docstring says what it covers and what it does not cover.

### Image comparison

- An image test returns the `Figure` and is decorated with `@pytest.mark.mpl_image_compare(savefig_kwargs={"dpi": 20})`, with the default tolerance.
  Existing tests with a higher `dpi` keep it until their baseline changes.
- When a plot changes, regenerate the affected baselines in the same commit.
  Explain each changed image in the commit message.

## Documentation

### Language

This applies to docstrings, comments, documentation pages, examples, release notes, commit messages and MR descriptions.

- Write simple English, for readers whose first language is not English.
  Prefer common words to rare ones. Use one term for one thing.
- Write short sentences: at most one main clause and one subordinate clause.
  Split longer sentences, or use a list.
- Write British English (`behaviour`, `colour`, `normalise`).
  Code, file names and quotations keep the spelling they have.

### Documentation pages

- Do not write what a reader assumes anyway: that a command works, is fast, or can be repeated.
- Write what would make the reader act wrongly without the sentence:
  a surprising wait, an effect beyond the command (a default that changes, files that are overwritten), a precondition, an option.

### Markdown files

- Write one sentence per line. Do not wrap a paragraph or a list item by length.

### Docstrings

- The summary line of a docstring ends with a period and is followed by a blank line.
- Document each parameter as `:param name:` and the return value as `:returns:`.
  Leave out `:returns:` only for `-> None`.
- Do not repeat types or defaults that are in the signature.
- Document the meaning, not only the name: units, reference frame, kind of tolerance (absolute or relative), allowed values, valid ranges, required data arrays.
- Keep docstrings, comments and example text true to what the code does.
  When you change what a function does, update its documentation.
- Describe the general behaviour.
  Do not single out a case that the general description already covers.
  The case belongs in a test.
- Mark inline code in docstrings with single backticks (`` `x` ``), in `.rst` files with double backticks.

### Gallery examples

For a package that documents its features in a sphinx-gallery:

- Show a new user-facing feature in a runnable example.
  Extending an existing example is often enough.
  Update the examples affected by an API change in the same MR.
- Name example files `plot_<name>.py`.
  When the reading order matters, prefix all files of the directory with a number with gaps (`plot_010_…`, `plot_100_…`).
- Start an example with a docstring: the title in sentence case, underlined with `=` of the same length, then one sentence on the purpose.
- Choose the thumbnail with `# sphinx_gallery_thumbnail_number = N` in a `sphinx_gallery_start_ignore` block directly after the docstring.
- Put all imports and global settings in the first code cell, without a heading.
- Mark text cells with `# %% [markdown]` and code cells with `# %%`.
  Above each code cell, explain what it does and why.
- Give sections a title at the highest heading level. Do not number the titles (`1.`, `I.`, `A.`).
- An example may check its result with `np.testing.assert_allclose` and an explicit tolerance.
  A comment says why this tolerance. The check is a code cell of its own, not hidden in a plotting helper.
- An example that compares against a known solution follows the order of the OGS benchmark notebooks:
  problem, parameters (a table of symbol, value and unit), analytical or reference solution, simulation, comparison, references.
- An example runs in at most 120 s in the docs build.
- Keep examples minimal:
  - show only the intended API,
  - keep intermediate results in named variables,
  - move helper logic into the library,
  - use `fig.show()` instead of importing matplotlib.
- Use the top-level API in examples.
  Inside the package, import from the specific submodule.
- A gallery directory has a `README.rst` (a title underlined with `=` and one or two sentences) and an entry in `subsection_order` in `docs/conf.py`.

### User guide

- Major functionality has a user-guide page, titled with the package name (`# logparser`).
  It has `## Overview` and `## Getting started`, lists the gallery examples and ends with the link to the API reference.
- The page separates usage, API and examples. It assumes little prior knowledge.

### Cross-references

- In gallery scripts (reStructuredText), reference API with roles and the full path: `` :meth:`~package.Class.method` ``.
  Link another example with `` :ref:`sphx_glr_auto_examples_<dir>_<file>.py` ``.
- In Markdown pages, use the MyST form `` {py:meth}`~package.Class.method` ``.
  Link an example with an empty-text link `[](../auto_examples/<dir>/<file>.rst)`.
- Use link text instead of bare URLs.
  Link external documentation only when it is official and maintained.
- The docs build without warnings with `-W --keep-going -n`.
  A necessary exception goes into `nitpick_ignore_regex` in `docs/conf.py`, with a comment.

## Packaging and files

- `requires-python` in `pyproject.toml` is the only place that defines the supported Python versions.
  The CI matrix follows it.

## Merge requests and commits

- An MR has one concern.
  Other improvements go into a follow-up MR or a linked issue, unless they shape the new API.
- An MR contains no unrelated changes, such as formatting churn or accidental reverts.
- Commit only maintained library content.
  Do not commit personal notes, application-specific notebooks, or generated files without their generator.
- The description says in a few sentences what the MR achieves.
  If the MR leaves something out or did not test it, the description says so.
- Apply a review point to all similar code, not only to the commented line.
- Discuss larger refactorings and design decisions with the core team first.
  The owner of the affected area reviews them.
- Once someone has reviewed the MR, do not rewrite the reviewed commits.
  Put later changes on top.
  You may reorganise the commits after the last review, until the next review.
- Before merge, after the final review:
  - rebase onto `main`,
  - squash `wip` and `fixup!` commits,
  - resolve all threads,
  - get a green pipeline on all platforms.
- After the subject, the commit message has one or two paragraphs:
  - first, one or two sentences on what changes and why,
  - optionally, one to four sentences with the reasons that the diff does not show.
- The length of the message follows the size of the change. A small change needs only a few words.
- Describe the state after the commit, in the present tense.
  Mention the earlier state only if the reader needs it to understand the change, and keep it short.
- Do not write that the change works; that is assumed.
  If a statement depends on a check, say how to repeat the check, not only its result.
- Use only numbers that stay true after the commit, such as a count that anyone can repeat.
  A run time depends on the machine and does not belong in the message.
- Link the primary source that the message relies on.
- Do not add `Co-authored-by` trailers, unless the author explicitly wants one for a specific commit.
- Keep one concern per commit. Separate refactoring and formatting from functional changes.
- A commit subject starts with its scope in brackets, as in OGS. Then it says what changes (`[meshseries] probe accepts XDMF series`).
  The scope is the affected subpackage in its exact spelling, or one of `docs`, `tests`, `ci`, `packaging`.
  The scope names the place, not the file. Do not invent a new scope for one commit.
- The forms `scope: subject` (`meshseries: probe accepts XDMF series`) and Conventional Commits `type(scope): subject` (`fix(meshseries): probe accepts XDMF series`) are accepted too.
  Do not mix the forms in one MR.
- A commit that changes behaviour states the new behaviour in the subject (`[meshseries] probe accepts XDMF series`).
  A commit that adds or fixes documentation uses the imperative (`[docs] Add coding standard`).
