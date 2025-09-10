# OGSTools 0.7.0 Release Notes

# Breaking changes

## API breaking changes

## Deprecations

### Examples

# Changes (non API-breaking)

## Bugfixes

## Features

- added `Meshes` class
- added `extract_boundaries` for simple extraction of boundary meshes from a 2D domain mesh
- added `plot.contourf_pv` for pyvista plots which work with `Variables`.
- `plot.contourf` now dispatches depending on the value of the new argument `interactive`:
  - None (default): 2D mesh -> matplotlib plot, 3D mesh -> interactive pyvista plot
  - True: always interactive pyvista plot
  - False: 2D mesh -> matplotlib plot, 3D mesh -> pyvista plot screenshot

## Infrastructure

- Added Binder links to example notebooks for interactive execution.

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release
