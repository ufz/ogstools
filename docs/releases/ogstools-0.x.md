# OGSTools 0.7.0 Release Notes

# Breaking changes

## API breaking changes

- removed the Mesh class entirely, MeshSeries is now a Sequence of native
  `pyvista.UnstructuredGrid`'s.
- `MeshSeries.scale` does not return a new MeshSeries, but changes the calling
  object.
- `MeshSeries.read_interp` -> `MeshSeries.mesh_interp`
- `MeshSeries.probe` -> `MeshSeries.probe_vals`
- `MeshSeries.extract_probe` -> `MeshSeries.probe`
- `MeshSeries.aggregate_over_domain` -> `MeshSeries.aggregate_spatial`
- `MeshSeries.aggregate_over_time` -> `MeshSeries.aggregate_temporal`
- `MeshSeries.plot_domain_aggregate` -> `ms.plot_line(variable.max/mean/...)`
- removed deprecated `MeshSeries.plot_probe` -> `probe = MeshSeries.probe(...); probe.plot_line(...)`
- `Matrix.mean -> Matrix.tensor_mean`
- `meshlib` has been renamed to `mesh` - classes to generate a new mesh (e.g.
  Layer, LayerSet, Surface, ...) have been moved to a submodule `create`.
- `ot.mesh.create.LayerSet.to_region_tetraeder` -> `ot.mesh.create.LayerSet.to_region_tetrahedron`

## Deprecations

### Examples

# Changes (non API-breaking)

## Bugfixes

- plot.line seems to have been missing the very first point of a linesample, now fixed

## Features

- MeshSeries
  - has now has difference method.
  - added support for logarithmic scaling in `ot.MeshSeries.plot_time_slice`.
  - new functions for renaming domain and subdomains
  - can now be scaled spatially / temporally in the Constructor
  - added `interpolate` method, to interpolate the MeshSeries data on a new mesh
  - added `compare` method to `ot.MeshSeries`.
- Meshes
  - save function performs partmesh if number of partitions are given, with optional dry_run
    - create_partitioning() and create_metis if vtu files are already present
  - subdomains() -> subdomain, domain_name() -> domain_name (with setter)
  - `plot` method displays domain mesh and subdomains (2D domain only for now)
  - `remove_material` method removes specified material id from domain and
    updates subdomains accordingly
  - `modify_names` method extends mesh names with prefix and/or suffix
  - `Meshes.from_mesh` now also works for 3D domain meshes
- Project
  - dependencies () return a list of referenced/needed files (meshes, xml includes, python scripts)
  - plot_constrains() -> overview plot with boundary conditions and source terms
- plot
  - add option to plot legend outside of plots.
  - added support for continuous contourplots via `continuous_cmap=True`.
  - Allow user to set arrowsize parameter in contourf
- Variable
  - added aggregation methods `min`, `max`, `mean`, `median`, `sum`, `std`, `var`.

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
