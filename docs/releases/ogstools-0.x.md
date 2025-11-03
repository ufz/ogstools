# OGSTools 0.7.0 Release Notes

# Breaking changes

## API breaking changes

- removed the Mesh class entirely, MeshSeries is now a Sequence of native
  `pyvista.UnstructuredGrid`'s.
- `MeshSeries.scale` does not return a new MeshSeries, but changes the calling
  object.

## Deprecations

### Examples

# Changes (non API-breaking)

## Bugfixes

## Features

- MeshSeries
  - has now has difference method.
  - added support for logarithmic scaling in `ot.MeshSeries.plot_time_slice`.
  - new functions for renaming domain and subdomains
  - can now be scaled spatially / temporally in the Constructor
- Meshes
  - save function performs partmesh if number of partitions are given, with optional dry_run
    - create_partitioning() and create_metis if vtu files are already present
  - subdomains() -> subdomain, domain_name() -> domain_name (with setter)
- Project
  - dependencies () return a list of referenced/needed files (meshes, xml includes, python scripts)
- plot
  - add option to plot legend outside of plots.
  - added support for continuous contourplots via `continuous_cmap=True`.
  - Allow user to set arrowsize parameter in contourf

## Infrastructure

- Gallery example figures are now tested against hashes which avoids
  unexpectedly changing figures.

## Documentation

### Tests

- plotting tests now actually check whether the resulting figure is as expected

### Imports

## Maintainer TODOs

### next sub release

### next main release
