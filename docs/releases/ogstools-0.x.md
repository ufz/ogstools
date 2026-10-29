# OGSTools 0.x Release Notes

# Breaking changes

## API breaking changes

## Deprecations

### Removed Modules

### Examples

# Changes (non API-breaking)

## Bugfixes

- the python files next to the python script (`<python_script>`) are now saved with it, so a Python BC script run by `Model.run()` can import them.
- `plot.contourf` failed to plot a variable on celldata with a mask, but is now able to.
- `Material.get_property` warns only once that it is deprecated ([!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510)).
- `Model.save` writes only the meshes that the model uses. Before, it linked the whole folder of the meshes, also with unrelated files ([!518](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/518)).
- `Result.save` with `overwrite=True` replaces an existing link instead of raising `FileExistsError` ([!518](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/518)).
- `ogsmonitor` logs `LogFileHandler` messages through the `ogstools.logparser` logger instead of printing them ([!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510)).
- Masking a variable no longer writes NaN into the mesh data.
- Eigenvalues and eigenvectors of masked tensors are NaN instead of raising an error or giving wrong values.

## Features

- variable aggregations over a meshseries can now be plotted directly e.g.:
  `fig = ot.plot.contourf(meshseries, ot.variables.temperature.max)`
- output names of variables aggregations have changed:
  - temperature\_{max|min|mean|median} -> {max|min|mean|median}\_temperature
  - temperature_std -> temperature_standard_deviation
  - temperature_var -> temperature_variance
- variables now have methods for derivatives (`gradient`, `divergence`, `vorticity`, `qcriterion`) which can be computed or plotted on the fly.
- meshseries now has a setitem method to set either data (`ms["new_data"] = ...`) or a mesh (`ms[-1] = ...` -> overwrites the last timestep)
- `MeshSeries.probe_values` now accepts integer ids in addition to coordinates and probes these very fast for Meshseries read from xdmf-files
- `MeshSeries.extract` and `MeshSeries.transform` now can now detach the originally read file to ensure consistency.
- added `Variable.from_total_stress(biot=..., bishop=..., pressure=...)` and `Variable.from_effective_stress(...)` to give a modified Variable, which converts between total and effective stress before evaluation of any other stored functions. Bishop's coefficient can be a constant or the name of a data field (cell data, e.g. from a material property, is interpolated to the points).
- `variables.dilatancy_critescu` and `variables.dilatancy_alkan` now assume effective stress and no longer read "pressure", so they behave the same as `variables.dilatancy_critescu_eff` and `variables.dilatancy_alkan_eff` as calculating from effective stress is the default for all OGS process (aside from TRM). For total stress input use e.g. `<variable>.from_total_stress(...)`.
- added `variables.fluid_pressure_criterion_clay` (== `variables.fluid_pressure_criterion`) and `variables.fluid_pressure_criterion_salt`
- The integrity criteria are masked by the field `variables.INTEGRITY_MASK` ("host_rock") instead of "displacement_active".

## Infrastructure

- `pytest-mpl` moved from the dependencies to the `test` extra, so installing ogstools no longer installs pytest.

## Documentation

### New Examples

- Evaluation of mechanical integrity criteria

### Updated Examples

- "Stress analysis" no longer has section for integrity criteria as they now have their own example.
- "Variable presets and data transformation" now explains a bit better how derived Variables work

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release
