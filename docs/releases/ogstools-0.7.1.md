# OGSTools 0.7.0 Release Notes

- Works with ogs==6.5.6
- Python: 3.10 - 3.13

## Breaking changes

## API breaking changes

## Deprecations

- Planned for `ot.meshes_from_gmsh` (replacement already available: `Meshes.from_gmsh`)

### Examples

## Changes (non API-breaking)

## Bugfixes

- `plot`: Fixed contour plots with very small value ranges (\< 1e-12) to display correctly
- `Project`
  - `run_model` : Fix execution of wrapper commands (e.g. source MKL)
  - Add the missing tag outputs to allow multiple output definitions
- Logparser: fix model_time analysis
- `MeshSeries`:
  - Fixed accessing meshes of extended MeshSeries from PVD files by ensuring timestep_files is updated correctly after extend().
  - Improved scale method to prevent crashes in complex, performance-intensive workflows; new implementation reuses cached meshes for better stability and memory efficiency.
  - fix extract probe for MeshSeries with a single point meshes
- `Mesh`:
  - Fixed crash in Mesh.difference() when meshes contain \_active datasets; computation now handles active field arrays
- `Feflow`:
  - Fixed handling of vector and tensor properties with NaN values (#135): tuples containing only NaNs are now dropped correctly, and tensor values are no longer misclassified

## Features

- Improved `BHE` meshing geometries as an input (by using shapely) - see !324
  - Supports georeferenced modeling in any metric CRS.
  - Flexible model area shapes (no rectangle restriction).
  - Prism mesh works for all cases; structured mesh for well-arranged setups.
  - Limitation: multiple refinement lines in one split surface not yet supported.
- Added `Meshes` class
  - `from_simulator` works directly with the OGS mesh in a running simulation
  - `from_gmsh` is `ot.meshes_from_gmsh`
  - `save` performs identify_subdomains
  - `from_msh`
    - Uses the newly introduced `extract_boundaries` for simple extraction of boundary meshes from a 2D domain mesh
- Added `plot.contourf_pv` for pyvista plots which work with `Variables`.
- `plot.contourf` now dispatches depending on the value of the new argument `interactive`:
  - None (default): 2D mesh -> matplotlib plot, 3D mesh -> interactive pyvista plot
  - True: always interactive pyvista plot
  - False: 2D mesh -> matplotlib plot, 3D mesh -> pyvista plot screenshot
- Added `to_polar` method for Variables. Useful, to convert stresses to a cylindrical or spherical coordinate system.
- Added `SimulationController` as wrapper for `OGSSimulation` to allow interruption and continuing simulations
- BHE
  - Allow ID notation

## Infrastructure

- Updated requirements!
- Added Binder links to example notebooks for interactive execution.
- Removed EXPERIMENTAL folder -> see Gitlab open issues
- Re-enable testing for Mac
- Faster pipeline execution (10min -> 6min)

## Documentation

- Added tutorials for interactive ogs simulation

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release

- deprecation notice for `ot.meshes_from_gmsh`
