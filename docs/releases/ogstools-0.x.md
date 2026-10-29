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
- `ogsmonitor` logs `LogFileHandler` messages through the `ogstools.logparser` logger instead of printing them ([!510](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/510)).
- Masking a variable no longer writes NaN into the mesh data.

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

## Infrastructure

- `pytest-mpl` moved from the dependencies to the `test` extra, so installing ogstools no longer installs pytest.

## Documentation

### New Examples

### Updated Examples

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release
