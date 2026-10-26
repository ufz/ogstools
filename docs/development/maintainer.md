# Maintainer Guide

# Release procedure

- Make sure there is a complete changelog at `docs/releases` and added to the corresponding `index.md`.
  - Migration guide for breaking changes
  - Links to merge requests
  - Include date of release
- Update pinned dependencies in `pyproject.toml` and `uv.lock`
- In `docs/conf.py` update the binder branch (around line 218) to the new ogs / ogstools tag pair. Update OGSTools version in the [binder repo](https://github.com/bilke/binder-ogs-requirements). See this [example](https://github.com/bilke/binder-ogs-requirements/commit/7a5bdd0847c7ae41f92a943d3b1c7c94ed704eea)
- Change fallback_version in pyproject.toml
- Create a release tag.
- Manual trigger publish job (for PyPI release)
- Wait for the tag pipeline to complete. This will also run a pipeline in [ogs/tools/ogstools-docs](https://gitlab.opengeosys.org/ogs/tools/ogstools-docs)-repo. After finishing check if the updated docs on ogstools.opengeosys.org are shown and the version selector is working.
- Create [GitHub release](https://github.com/ufz/ogstools/releases) -> a Zenodo release is automatically created.
- Update authors on Zenodo release.
- Update Zenodo badge in repo.
- On https://github.com/conda-forge/ogstools-feedstock create a new issue with the title `@conda-forge-admin, please update version`
- Search the codebase for `TODO(ogs-version)` and simplify any now-obsolete ogs-version-dependent code/tests.

# Post-release procedure

- Update the ogstools version in OGS (Benchmarks) in one MR: the pin `ogstools==` in `Tests/Data/pyproject.toml`, `binder_tag` in `Tests/Data/testrunner.py` (or wherever the pin's comment points) and `$tools_url` in `web/layouts/_partials/header.html`
- Discourse announcement

# Coding standard

These notes are for the people who write and update the [coding standard](coding-standards/index.md).

## Sources of the rules

- The rules come from the review discussions of the merge requests on GitLab and from the conventions that the code on `main` follows.
- Where the two disagree, the most recent decision of the core reviewers wins.
  Where reviews say nothing, the convention that most of the code follows wins.
- A rule that deviates from the conventions of OGS, pyvista or Python best practice gives its reason here.

## How the pages are built

- The checklists and the patterns state only the current or wanted rules, without history and without reasons.
  Reasons go into the list below.
- A rule is worded no stricter than its reason. Use "prefer" where another way is also fine.
- A rule goes into `shared.md` when it holds for every Python project around OGS, and into `ogstools.md` when it holds for OGSTools only.
  `shared.md` names OGSTools only as the shared tool for running OGS, reading meshes, units and plots.
  An OGSTools example of a shared rule goes into `ogstools.md`.
- The checklists list only rules that no tool checks.
  When a tool takes over a rule, remove the rule from its checklist.
  The commit rules stay in the checklists, because commits made in the GitLab web interface skip the commit-msg hook.
- A new tool rule starts with a baseline: the files that violate it at that time.
  Generate the baseline when you enable the rule, in `[tool.ruff.lint.extend-per-file-ignores]` or in the `exclude:` of the hook.
- `AGENTS.md` links to the coding standard and adds only what applies to agents. It does not repeat rules.
- In the checklists, `→` links to the pattern that gives the full rule.

## Reasons behind rules

Each reason links the review threads where the rule was raised, as `!<number>`.
A reason without such a link has no review thread that a keyword search of the threads of all MRs found, with comments of review bots left out.
It follows from the linked documentation or from how the code works.

### How the standard is applied

- The rules apply to new and changed code, so that an MR does not grow with unrelated fixes. A sweep over existing code is its own MR [!273](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/273).
- Rules that a tool can check go into the tool, and the checklists keep only the rest [!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510).
  A tool gives the same answer every time and needs no reviewer.
- A baseline lets a new tool rule start without fixing all old code first. New code is checked in full, and the list only shrinks.
- CI runs the same hooks as `pre-commit run --all-files` on a developer's machine, so a green local run means a green CI run.
- Where no rule applies, the conventions of OGS come first, because OGSTools is part of the OGS project.
- The review process is that of OGS: the `workflow::please review` label, a review by a core developer, approval and a green CI ([OGS code reviews](https://www.opengeosys.org/docs/devguide/development-workflows/code-reviews/)).
- A review comment that names its rule can be checked against the standard. It stays a proposal, because decisions stay with people [!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510).
- The coverage marks in the MR diff show untested lines without a separate report [!473](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/473).
- An MR is merged without squashing, so that the reviewed commits stay as they are.

### Shared rules: simple code

- Flat code is easier to read than nested `if`/`else` [!31](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/31), and an early `return` shows the special case first.
  A helper at module level, not a nested `def`, can be profiled with line_profiler and keeps the nesting low [!278](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/278).
- A comprehension builds a collection in one expression.
  A plain loop shows that the body changes something else, such as an existing object or a file [!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510) [!158](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/158) [!432](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/432).
- Vectorised numpy is faster than a Python loop over an array and shorter to read.
- `if x:` treats `0` and `""` like `None`, which is wrong where they are valid values. [PEP 8](https://peps.python.org/pep-0008/#programming-recommendations) asks for `is None`.
- An argument that equals the default repeats information that the signature already has, and it goes stale when the default changes [!396](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/396).
- A named number says what it is, and a comment says why it has this value [!324](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/324).
- A small function can be tested and reused on its own. A long test is split into independent tests [!148](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/148).
- Code that already exists is tested and known to reviewers. Reviewers point to it in many MRs [!483](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/483) [!418](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/418) [!423](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/423).
- A shared helper with one caller is speculation. It is added when a second caller needs it.
- A fix at one caller leaves the other callers broken. Reviewers ask to check all other occurrences [!419](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/419) [!114](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/114) [!77](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/77).
- OGSTools finds the OGS binaries, checks them and logs their output. A direct `subprocess` call skips this.
- A unit conversion by hand is easy to forget on one code path. A missing `/ 86400` on one branch was a bug [!77](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/77).
  `Variable` and `u_reg` convert in one place and label the result.

### Shared rules: names

- A name that says what a value is saves a comment. Unknown abbreviations make readers guess [!88](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/88) [!62](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/62).
  Single letters are fine for loop indices and for math symbols that the formula uses [!119](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/119); reviewers do not want very long names for short-lived locals [!173](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/173).
- `_temp` or `_tmp` says nothing about what the value holds.
- A plural name tells that an argument is a collection [!128](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/128).
- A boolean that reads as a predicate (`is_saved`) makes `if` lines read like sentences [!416](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/416) [!407](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/407).
- `get_` and `set_` are reserved for methods. A free function is named after what it returns [!158](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/158) [!374](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/374) [!62](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/62).
- `read` and `save` mean file input and output. A function that does not touch files gets another verb [!78](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/78).
- A name that promises several values must return several [!432](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/432).
- `from_<source>` is the pattern users know from the file-backed objects [!477](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/477) [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342).
- `snake_case`, `CapWords` and `UPPER_CASE` are the names of [PEP 8](https://peps.python.org/pep-0008/#naming-conventions) [!450](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/450) [!173](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/173).
- One term per concept lets users search for it. The review checklist asks to check the introduced names against the existing ones and against general conventions [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342) [!417](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/417).
- A module name repeated in a function name (`mesh.mesh_read`) makes the call longer and says nothing new.

### Shared rules: public API

- Everything that `__all__` exports is a promise to users. A leading `_` marks the rest as internal, so it can change [!330](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/330) [!62](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/62).
- Public API needs documentation, tests and an example. Making a private function public later is easy; the reverse breaks users [!428](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/428) [!452](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/452) [!444](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/444) [!432](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/432) [!430](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/430).
- Users pass what they have at hand, such as a string for a path or a name, and should not have to import `pathlib` or look up the object for a quick call [!355](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/355) [!437](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/437).
  Internal helpers get one type and need no checks.
  Users pass lists and tuples where an array is expected, and should not have to convert them [!284](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/284) [!245](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/245).
- `Sequence` accepts tuples and other sequences, so the hint does not reject valid input [!376](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/376).
- An opaque tuple makes users unpack and remember the order [!333](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/333).
  A named type documents its fields [!159](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/159).
- Users learn one sibling function and expect the same signature from the others.
  `append` and `extend` change the object and `__add__` returns a new one, as in Python itself.
- Changing an argument is a side effect that the caller does not see [!71](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/71) [!255](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/255).
  "`x.fun(y)` should almost never mutate `y`" [!255](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/255). If a function changes its input, its docstring says so [!78](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/78).
  Copy first, with a new name [!468](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/468).
- `copy(deep=False)` shares data with the original, so a change to one shows up in the other. A comment says why that is safe.
- A separate function per input kind, a named option and a required argument each make the call say what happens.
  A wrong default that silently gives a wrong result is worse than an error.
- Typed parameters show up in the signature, in the docs and in the IDE. `**kwargs` is for niche options and large option sets such as plot styling [!339](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/339).
  Reviewers prefer specific keyword arguments while their number and combinations stay small [!458](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/458).
- A default of `None` lets the function read the current configuration at call time.
  A configuration value in a default is read once at import, and a later change is ignored.
- A simpler interface is easier to use and to keep stable. It can be extended when a use case appears [!452](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/452) [!134](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/134).
- A dataclass, enum or `NamedTuple` documents its fields and lets mypy check them. A nested dict does not [!432](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/432) [!159](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/159).
  A frozen dataclass cannot be changed by accident [!259](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/259).
- A read-only property cannot get out of sync with the data it is derived from.
- A function that writes files or prints cannot be reused where the output is not wanted. Writing, printing and argument parsing stay at the edges, in `save` and in command-line tools.
- A function that reads a mutable module-level object depends on what other code did before. Mutable global variables caused tests to fail at random [!322](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/322).
- A library must not change the logging configuration of the application that imports it. The [Python logging HOWTO](https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library) asks libraries to add no handlers.
- Users should not need to know the file format. The same code works for PVD and XDMF.
- Plot styling read at call time follows changes that users make to `ot.plot.setup` [!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510).
  `Variable.get_label` gives every plot the same label form.
- Keyword-only options are allowed, but not required. They can be reordered without breaking user code, and positional options keep short calls short [!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510).

### Shared rules: changing existing API

- Users and the OGS benchmark notebooks depend on the API. A break costs every user a change.
  A patch release promises not to break the API; a new parameter goes at the end [!483](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/483).
- Code that is moved or merged must not lose its input checks. A lost check turns a clear error into a confusing one later.

### Shared rules: typing

- `Literal` lets mypy and the IDE check the allowed options [!255](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/255) [!175](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/175).
  A hint that is narrower than the accepted input rejects valid calls [!437](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/437).
- `-> Self` keeps the return type right in subclasses. It needs `typing_extensions` while Python 3.10 is supported [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342) [!376](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/376).
- One order (`Path | str`) makes signatures easier to compare.
- `from __future__ import annotations` makes all annotations lazy, so no annotation needs quotes.
- A bare `# type: ignore` hides the next real error on that line. Reviewers ask to fix the typing instead [!249](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/249) [!158](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/158).
- `pv.UnstructuredGrid` is the mesh type of OGS output. `pv.DataSet` tells mypy that any dataset is accepted.

### Shared rules: errors, warnings and logging

- An error at the entry point names the user's input. The same error deep inside names an internal value and confuses users.
- A final `else` that raises catches cases that nobody thought of [!402](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/402) [!483](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/483).
- Built-in exceptions are what Python users expect to catch. A custom exception derives from the fitting one, so that `except ValueError` still works.
- Python removes `assert` under `python -O`, so an `assert` cannot check user input [!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487). `raise UserWarning(...)` raises a warning class as an exception, which callers do not expect.
- A message with the wrong value and the remedy lets users fix the call without reading the code [!386](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/386) [!311](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/311) [!402](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/402).
  A warning can also say how to avoid it [!416](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/416).
- `warnings.warn` and `logger.warning` are split as in the [Python logging HOWTO](https://docs.python.org/3/howto/logging.html#when-to-use-logging) [!77](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/77) [!289](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/289).
- A module-level logger named after the module lets users turn on the messages of one subpackage [!359](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/359).

### Shared rules: comments

- The code shows how; only a comment can say why [!34](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/34) [!382](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/382) [!289](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/289).
  A comment gives its why in one clause, because the reader of the line needs only that. Longer reasoning goes into the developer guide, where readers look for it.
- A comment that repeats the code goes stale when the code changes [!34](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/34) [!77](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/77). A wrong comment misleads more than none [!321](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/321) [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342).
- A date in a comment does not say whether the statement is still true. A commit or MR can be looked up.
- A `TODO` with an issue link is tracked. A `TODO` in a docstring shows up in the rendered docs [!477](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/477) [!483](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/483).
- Unused code still has to be read and maintained [!461](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/461) [!249](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/249) [!173](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/173).
- The `TODO(ogs-version)` tag lets maintainers find the workarounds that a new minimum OGS version makes obsolete.

### Shared rules: imports and dependencies

- An absolute import shows where a name comes from [!53](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/53) [!273](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/273). Relative imports in `__init__.py` keep the export lists short.
- The command-line tools and the OGS notebooks import the package often. Heavy imports at module top slow down every start [!304](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/304) [!485](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/485).
- A package that is imported but not declared works only by chance, through another dependency [!282](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/282).
  A lower bound states the oldest version that works. An upper bound blocks users from fixes and is hard to remove, see [Should you use upper bound version constraints?](https://iscinumpy.dev/post/bound-version-constraints/).
- Test and docs tools are not needed to use the package. Their extras are installed only by developers.
- Every dependency must be installed, updated and trusted by every user.
- A non-core dependency in an extra keeps the default install small. The install hint tells users how to get it.
- Copied code is not updated with its source. A fix upstream helps all users of that library.
- The project has one license. Extra license files raise questions about which license applies [!13](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/13) [!6](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/6).

### Shared rules: tests

- Untested code breaks without notice. Documentation examples do not count, because they are not part of the coverage report [!366](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/366) [!473](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/473).
- Every supported input variant needs a test, because each takes its own code path, for example PVD and XDMF [!359](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/359) [!314](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/314).
- A failure case shows that invalid input gives a clear error and not a wrong result.
- A flat `tests/` folder and one module per subpackage make tests easy to find.
- Plain functions with fixtures are the pytest style. Test classes add a level that new tests do not need.
- A fixture next to its test is easy to find. pytest picks up shared fixtures only from `conftest.py` [!398](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/398).
- A copied test differs from the original only in its data. A parametrized test shows that difference [!314](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/314) [!417](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/417) [!402](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/402) [!431](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/431).
  Readable `ids=` keep the image baseline file names stable.
- Extending an existing test keeps related checks together. The review checklist asks for this [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342) [!417](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/417), and reviewers ask to test a function in its existing test [!383](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/383).
- `np.testing.assert_allclose` reports which elements differ, a plain `assert` only that something differs [!71](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/71).
  An explicit tolerance says what precision the test expects [!155](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/155) [!376](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/376).
- An assertion on a concrete value fails when the result is wrong. A `try`/`except` around a test can hide every failure.
- `tmp_path` gives each test its own folder, so tests do not overwrite each other [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342).
  `monkeypatch` resets global state after the test, so tests can run in parallel [!407](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/407).
- A `reason=` says why a test does not run, so that someone can remove the skip later.
- Small test data keeps the suite fast [!276](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/276). Data generated by a script can be checked and changed.
- Fast tests are run more often. pytest reports the tests over 10 s.
- A keyword argument at the call shows the meaning of a value such as `True`.
- Markers let developers skip the tests that need OGS or its tools [!392](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/392).

### Shared rules: test docstrings

- The name or the docstring of a test must say what is under test [!358](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/358).
  The docstring says what the test is for, because a generated docstring often describes the proxy check and not the purpose [!432](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/432).
- The test levels are those of [ISTQB CTFL v4.0](https://astqb.org/2-2-test-levels-and-test-types/).
  The oracle categories build on the survey by [Barr et al. (2015)](https://doi.org/10.1109/TSE.2014.2372785); `implicit` is their term, and `consistency` is a pseudo-oracle in their terms.
  The unit/component split was dropped because the code never used it [!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510).
- Tolerances in the docstring go stale when the code changes.
- A sentence about other tests in a test docstring becomes false when those tests change.

### Shared rules: image comparison

- A low `dpi` keeps the baseline images small and still shows layout changes.
- A baseline regenerated in the same commit as the plot change shows the reviewer the old and new image side by side [!358](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/358) [!402](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/402).

### Shared rules: documentation

- Many users and developers do not speak English as their first language. Simple words and short sentences are easier to read and to translate.
- British English is the spelling of the project's text. Code keeps the spelling of the libraries it calls (`color` in matplotlib).
- A sentence that says what a reader assumes anyway costs reading time. A sentence that prevents a wrong action is worth it.
- One sentence per line lets search find a phrase that would otherwise cross a line break, and a diff shows the sentences that changed. OGS asks for the same, so that changes are easier to review and track ([OGS documentation guidelines](https://www.opengeosys.org/docs/devguide/documentation/docs-guidelines/)).

### Shared rules: docstrings

- Each parameter needs its meaning. Reviewers ask for missing `:param:` and `:returns:` fields [!390](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/390) [!432](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/432) [!271](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/271) [!114](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/114).
- Types and defaults are in the signature. Repeating them adds nothing and goes stale [!304](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/304) [!363](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/363) [!396](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/396).
- Units, tolerances and valid ranges are what users get wrong without documentation [!376](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/376) [!324](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/324).
- A docstring that disagrees with the code misleads users [!428](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/428) [!278](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/278) [!369](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/369).
- A docstring describes the general behaviour. A special case that an MR just fixed belongs in a test.
  Not "The reader is chosen by the file extension of `filename`, also when `filename` is a symbolic link to a file without extension", but "The reader is chosen by the file extension of `filename`".
  Details of the change go into the commit message or the MR description [!477](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/477).
- Single backticks render as code through `default_role = "code"` in `docs/conf.py`.
  `.rst` files keep double backticks, because the `rst-backticks` hook requires them.

### Shared rules: gallery examples

- Users learn a feature from its example. A public feature needs one [!452](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/452).
- Numbers with gaps leave room to insert an example later without renaming the others.
- The rules on imports, headings and numbered titles follow the example guidelines that the developer guide has had since 2024, "to maintain consistency in style and structure between different examples" [!201](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/201).
- The thumbnail number must be set where sphinx-gallery reads it, and hidden from readers [!119](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/119).
- A check in its own cell shows the reader what is verified. Hidden in a helper, it looks like plotting.
- Examples that compare with a known solution follow the OGS benchmark notebooks, so users of both find the same structure.
- The docs build runs every example. A slow example slows down every pipeline.
- `fig.show()` removes the need to import matplotlib only to show a figure [!480](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/480).
- Examples show the API that users call. The package itself imports from the submodules [!199](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/199) [!369](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/369), which also avoids circular imports through the top-level `__init__.py`.
- The `README.rst` and `subsection_order` give each gallery a title and a fixed place in the navigation.

### Shared rules: user guide and cross-references

- A user guide with little required knowledge is the entry for new users [!173](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/173). This follows the [OGS documentation guidelines](https://www.opengeosys.org/docs/devguide/documentation/docs-guidelines/): keep the entry level low.
- Roles with the full path are checked by Sphinx and break the build when an API is renamed.
- Link text says where a link goes. Official documentation stays available.
- With `-W`, a broken reference fails the build and is fixed in the MR that caused it.

### Shared rules: packaging and files

- One place for the Python versions avoids settings that disagree, such as the ruff target version [!153](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/153).

### Shared rules: merge requests and commits

- An MR with one concern is easier to review and to revert. Other topics go into a follow-up MR or issue [!427](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/427) [!464](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/464) [!452](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/452) [!511](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/511) [!477](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/477).
  A clean history does not mix the changes of two MRs [!276](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/276).
- Personal notes and one-off notebooks are not maintained. A generated file without its generator cannot be updated.
- The description tells reviewers what to look for and what was left out.
- A review point usually applies to similar code elsewhere [!419](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/419) [!114](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/114) [!77](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/77).
- A larger refactoring touches code that others own. It is discussed first with the people involved [!432](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/432) [!251](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/251).
- Reviewed commits stay unchanged, so that reviewers can check out the state they reviewed and see what changed since.
- `fixup!` commits are squashed before merge [!19](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/19), so that the history on `main` has one commit per change.
- A commit message is read by someone who looks for why the code is as it is. Reviewers ask for descriptive messages [!404](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/404) [!455](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/455), for the source of copied data [!427](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/427), and for what a data file contains [!504](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/504).
- A statement that a change works says nothing. How to repeat a check does.
- A run time differs between machines, so it is not true for the next reader.
- A separate refactoring commit can be reviewed by checking that nothing changed. A test that fails first and a fix that turns it green show the bug [!393](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/393).
- The `[scope]` form is the form of OGS commits ([OGS history](https://gitlab.opengeosys.org/ogs/ogs/-/commits/master)). OGS uses `scope:` and Conventional Commits too, so all three are accepted.
  The scope tells reviewers which part of the package changes [!455](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/455).

### OGSTools rules

- "OGSTools" is the name of the project in text. `ogstools` is the name of the package, the import and the command.
- `Variable | str` lets users write `"temperature"` instead of looking up the preset. `Variable.find` resolves the name in one place [!259](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/259) [!158](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/158) [!483](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/483).
- `ogstools._internal.deprecated` gives every deprecation the same message [!417](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/417) [!348](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/348).
  A deprecated function gets no new features, because it will be removed [!277](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/277).
- The OGS benchmark notebooks are the largest users of OGSTools. `OGSTOOLS_REF` runs them against a branch in OGS CI, as described in the [OGS CI documentation](https://www.opengeosys.org/docs/devguide/testing/gitlab-ci/).
- The listed packages each take long to import, and most users need only some of them [!304](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/304). lxml is needed by the project-file classes, which most users load.
- A path built from `__file__` breaks when a file moves. `ogstools.definitions` defines the paths once [!477](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/477) [!146](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/146) [!452](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/452).
- Test data loaded through `ogstools.examples` is shared by tests, examples and users [!62](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/62).
- `hypothesis` tests find edge cases that hand-picked values miss [!283](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/283) [!329](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/329). A limit on the input range needs a comment that says why [!283](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/283).
- The `system` and `tools` markers let developers skip tests that need OGS [!392](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/392). Their time limits keep the slow test jobs short.
- `make test_figures` selects the image tests by the names `TestPlotting` and `test_plot`, so other names get no baseline.
- Examples write to `ot.definitions.temp_dir`, not to `OGS_TESTRUNNER_OUT_DIR` as the OGS benchmark notebooks do, because sphinx-gallery runs them, not the OGS test runner.
  A fixed folder in the temporary directory would be overwritten by other examples [!123](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/123) [!450](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/450).
- Figure rendering depends on fonts and graphics libraries of the system. Gallery hashes are therefore made in the CI image, where they are checked.
- The release notes are the only place where users learn about a change. A breaking change without migration example leaves users to find the fix themselves [!363](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/363) [!348](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/348) [!241](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/241) [!344](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/344).
  Inline code in release notes stays readable on GitLab and on GitHub.
- The commit scopes match the subpackages, so that `git log` can be filtered by scope.

### Patterns

- One skeleton per family means users learn one way to construct, save and copy all objects of a kind [!477](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/477) [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342).
- A file-backed object is bound to its file. `copy()` gives an independent object, so changes do not reach the source [!477](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/477) [!253](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/253).
- `save` has the same parameters in every class, so that saving at the highest level behaves the same as below [!342](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/342).
- A classmethod constructor must return a new object of its class [!173](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/173).
- A constructor that writes files would surprise users and break `dry_run`. Writing happens only in `save`.
- `__repr__` that can be evaluated lets users copy an object into code.
- Objects that compare by mutable content must not be hashable, as the [Python data model](https://docs.python.org/3/reference/datamodel.html#object.__hash__) requires.
- A command-line tool that only maps arguments keeps all logic testable in the library.
  The `--help` test catches entry points that point to a module that does not exist, as `yaml2vtu` did.
- Plot functions take the data first and the variable second, as pyvista and matplotlib take the data first.
  A figure-level function returns its figure, so that a notebook shows it and a test can compare it.
- `output_file` is one argument for path and file name [!190](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/190) [!199](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/199).
- A generator returns the path of the written file, a reader the object, so that both can be chained. A function that writes a file with a hard-coded name cannot be reused [!306](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/306).
- A loader returns a new object, so that one test cannot change the data of another [!253](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/253).
  Presets and example data use the same names and units [!322](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/322).
- `__str__` and `__repr__` start from the base class, so that the storage state shows the same way for every class.
- `Result` and `Simulation` have no `id` in the constructor, because they only exist after a run.
  `ReferencedFile` objects belong to their `Project`, so users never construct them.
- The compatibility constructors keep old user code working. New classes do not add more of them, so there is one way to load from a path.
- `argparser()` returns the parser, so that a test can check the options without running the tool.
  `cli()` returns the exit code, so that a test can call it without ending the process.
- `-l/--log` is the same switch in every tool. Logging is configured in `cli()`, not on import, see the shared rule on global state.
- One tool per subpackage in `_cli.py` makes each tool easy to find, and the leading `_` keeps it out of the public API.
- A new axes-level function stays internal, so that users meet only figure-level and report-level functions.
- `BuildTree` components work on the shared XML tree of their `Project`, so a change through one attribute is seen by all others.
- A record's fields follow the regex groups, because the parser fills them in that order.
  A raw-string pattern with escaped literal characters matches the log line exactly.
- An analysis checks its input and output columns, so a log of another OGS version fails with a clear message.
- A loader calls the public constructor, so that the example data goes through the same code as user data.
- A path constant with the file kind as prefix (`prj_`, `log_`, `msh_`) shows in completion what kind of file it is.
- `add_` and `set_` return `None`, as Python's own mutating methods do (`list.append`).
- Two unit registries cannot be mixed: pint refuses to combine quantities from different registries.
- One regex table per log-format version keeps old logs readable after OGS changes its log format [!312](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/312).

### Tool configuration

- ruff does not check `docs/` for commented-out code, because gallery scripts show code in `code-block` directives and set the thumbnail with a `sphinx_gallery_thumbnail_number` comment.
- Rules that a tool can check go into the tool, and the checklists keep only the rest [!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510).

## Open points

- Lazy loading of subpackages and the rules on heavy imports wait for the lazy-import refactoring ([!485](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/485)).
  Then ruff can check imports at module level and imports only needed for type hints.
- ogsworkflows links to `shared.md` and adds its own page. A change to `shared.md` therefore goes through an OGSTools MR.
  Open: whether `shared.md`, the shared hooks and `scripts/check_ruff_baseline.py` move into a shared repository.
  Projects would then pin a version of its hooks and copy its ruff settings, since ruff cannot read settings from a URL.
- The `pinned` extra and `requirements/*.txt` are removed with the next major update (0.9 or 1.0).
- Candidates for a tool check: absolute imports, the docstring summary line, complete `:param:` fields (pydoclint), naming, `Path | str` order, reasons for `# type: ignore`, `-> Self` for alternative constructors, `uv lock --check` and declared dependencies (deptry).
