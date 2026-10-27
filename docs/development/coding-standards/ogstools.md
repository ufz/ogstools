# OGSTools

These rules apply only to OGSTools, in addition to the [shared rules](shared.md).
The [patterns](patterns.md) describe the framework skeleton that the components of OGSTools follow.

## Names

- Write "OGSTools" in text. Write `ogstools` only for the package, an import or a command.

## Public API

- Variables and units follow their skeleton. → [Variables and units](patterns.md#variables-and-units)
- File-backed classes follow the storage skeleton. → [File-backed objects](patterns.md#file-backed-objects)
- Command-line tools follow their skeleton. → [Command-line tools](patterns.md#command-line-tools)
- `**kwargs` is also used for the open-ended parts of the OGS XML schema. → [Project-file builders](patterns.md#project-file-builders)

## Changing existing API

- Mark replaced API with `ogstools._internal.deprecated`, which warns with a `DeprecationWarning`.
  Add no features to deprecated functions. Use the current API in new code, tests and examples.
- Every breaking change gets a release-notes entry with a migration example (see [Release notes](#release-notes)).
- The OGS benchmark notebooks (`Tests/Data/**/*.py` in an OGS checkout) use OGSTools, mostly `ot.Project`, `ot.MeshSeries`, `ot.plot.contourf`, `ot.plot.line` and `ot.variables`.
  Before a change to API they use is merged, start the manual job `ogs/ogs-notebooks` in the MR pipeline.
  It sets `OGSTOOLS_REF` to the MR commit and runs the notebooks in OGS CI, see the [OGS CI documentation](https://www.opengeosys.org/docs/devguide/testing/gitlab-ci/).
  Before you remove a deprecated function, search the notebooks for it.
  If they still call it, update them first in an OGS MR.

## Imports and dependencies

- Import matplotlib, pyvista, pandas, scipy, gmsh, bokeh, ogs and jupytext inside the function that needs them. lxml may be imported at module top.
- Build package paths from `ogstools.definitions` (`ROOT_DIR`, `EXAMPLES_DIR`), not from relative paths.

## Tests

- Load data through the loaders and constants in `ogstools.examples` (`examples.load_meshseries_THM_2D_PVD()`, `examples.prj_mechanics`).
  Add a new one there. Do not build an `EXAMPLES_DIR` path where you call it.
- Think of a property-based test with `hypothesis` (`@given`) when a function accepts a wide range of inputs:
  geometry, mesh sizes, units, parsers, round trips like `save` then `read`.
  The profiles in `tests/conftest.py` set the number of examples and the deadline. Override them in a test only with a comment why.
- Mark a test that runs a simulation with `@pytest.mark.system`.
  Mark a test that calls an OGS binary tool with `@pytest.mark.tools`.
- Keep a test marked `tools` within 20 s, and a test marked `system` within 60 s.
- An image test lives in `tests/test_plot.py`, or its name starts with `test_plot_`.

## Documentation

### Gallery examples

- `docs/_build/html/sg_execution_times.html` lists the run times of the examples.
  For a longer simulation, coarsen the model or load a precomputed result from `ogstools.examples`.
- An example lives in `docs/examples/howto_<topic>/plot_<name>.py`.
- Use the top-level API (`ot.MeshSeries`, `ot.Project`). Load data with `from ogstools import examples`.
- Write example output to `ot.definitions.temp_dir("<example_name>", "examples")`.
- When the figure of an example changes on purpose, regenerate `docs/gallery_hashes.json` in the same MR.
  Regenerate it in the CI devcontainer image, see [Gallery hashes](../index.md#gallery-hashes).

### Release notes

- Every user-visible feature, fix and breaking change gets a release-notes bullet, under its fixed headings.
  The bullet ends with the MR link. Internal changes get no bullet.
- Name API in release notes as inline code with its class (`` `Class.method` ``), not as Sphinx roles.
- Release-notes bullets go into `docs/releases/ogstools-0.x.md`.
- The MR link at the end of a release-notes bullet has the form `([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))`.

## Merge requests and commits

- Commit scopes are the subpackages of OGSTools in their exact spelling: `mesh`, `meshes`, `meshseries`, `plot`, `logparser`, `variables`, ….
