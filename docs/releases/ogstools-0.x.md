# OGSTools 0.x Release Notes (upcoming release)

This is not released yet!

# Breaking changes

- OGS Wheel (`pip install ogs`) is now an optional (before mandatory) dependency of OGSTools. OGSTools requests either a OGS wheel OR a custom OGS made available on PATH or OGS_BIN_PATH.

## API breaking changes

- mesh.read_shape (removed function, functionality is not covered anymore)

### Examples

- removed all examples with shape files

# Changes (non API-breaking)

## Bugfixes

## Features

- new logparser analysis to inspect simulation behaviour over clock time and model time
- new logparser plot to create an overview of the convergence behavior for the
  entire simulation (ot.logparser.plot_error, ot.logparser.plot_convergence_order)
- OGS simulation can be run in a background process via ot.Project.run_model(..., background=True)
- MeshSeries.extract_probe/probe/values accept str/Variables and lists of them as arguments (with improved performance)

## Infrastructure

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release

- MeshSeries: from_data() constructor -> __init__()
