# OGSTools 0.8.0 Release Notes

# Breaking changes

## API breaking changes

### New Core Framework and Storage System

- **New unified framework**: Introduced `Model`, `Simulation`, and `Result` classes
  that provide a complete workflow from setup to analysis. See the new
  `plot_framework_short.py` example for a quick overview.
- **Storage system**: All major classes (`Model`, `Meshes`, `MeshSeries`, `Project`)
  now inherit from `SaveBase` and support:
  - Unified `save()` / `from_folder()` / `from_id()` interface
  - Automatic backup on overwrite (configurable via `SaveBase.Backup`)
  - Archive mode for creating self-contained copies
  - ID-based organization in user-defined storage paths
- **Simulation controllers**:
  - Removed deprecated `ogstools.simulation` module
  - New `SimulationController` base class with `OGSInteractiveController` and
    `OGSNativeController` implementations
- **Log parsing**: Added `Log` class for parsing and analyzing OGS log files with
  methods for convergence analysis and simulation status tracking

### MeshSeries and Meshes

- removed the Mesh class entirely, MeshSeries is now a Sequence of native
  `pyvista.UnstructuredGrid`'s.
- `MeshSeries.scale` changes the calling object and returns it.
- `MeshSeries.read_interp` -> `MeshSeries.mesh_interp`
- `MeshSeries.probe` -> `MeshSeries.probe_vals`
- `MeshSeries.extract_probe` -> `MeshSeries.probe`
- `MeshSeries.aggregate_over_domain` -> `MeshSeries.aggregate_spatial`
- `MeshSeries.aggregate_over_time` -> `MeshSeries.aggregate_temporal`
- `MeshSeries.plot_domain_aggregate` -> `ms.plot_line(variable.max/mean/...)`
- removed deprecated `MeshSeries.plot_probe` -> `probe = MeshSeries.probe(...); probe.plot_line(...)`
- `MeshSeries.save()` signature changed: now uses unified storage interface with
  `target`, `overwrite`, `dry_run`, and `archive` parameters
- `Meshes` now supports storage operations: `save()`, `from_folder()`, `from_id()`
- `Meshes.__init__()` now accepts optional `id` parameter
- `Meshes` implements `__eq__()` and `__deepcopy__()` for proper comparison and copying

### Project

- `Project.__init__()` signature changed:
  - `output_file` is now optional (defaults to auto-generated path if using IDs)
  - Added `id` parameter for storage system integration
- `Project` now inherits from `SaveBase` and supports `save()`, `from_folder()`,
  `from_id()` operations
- `Project.inputfile` renamed to `Project.input_file`
- `Project` implements `__eq__()`, `__deepcopy__()`, `__repr__()`, and `__str__()`

### Other API Changes

- `Matrix.mean` -> `Matrix.tensor_mean`
- `meshlib` has been renamed to `mesh` - classes to generate a new mesh (e.g.
  Layer, LayerSet, Surface, ...) have been moved to a submodule `create`.
- `ot.mesh.create.LayerSet.to_region_tetraeder` -> `ot.mesh.create.LayerSet.to_region_tetrahedron`
- `ot.mesh.create.dataframe_from_csv` doesn't require a parameters .csv anymore,
  but either a mapping of layer_id to surface files, or the directory containing
  all surfaces files (in that case the layer_ids map to the sorted file list)
- `ot.mesh.geo.depth` now requires top_mesh as an argument.
- `ot.mesh.geo.p_fluid` was removed. See the `Stress analysis` example for how
  to calculate fluid pressure now.
- `ot.mesh.file_io.save()` signature changed: mesh parameter now comes first

## Deprecations

### Removed Modules

- Removed `ogstools.simulation` module (replaced by `ogstools.core` framework)

### Examples

# Changes (non API-breaking)

## Bugfixes

- plot.line seems to have been missing the very first point of a linesample, now fixed
- cli dashed arguments now work as expected

## Features

### New Core Framework

- **Model**: New class combining project file, meshes, and execution settings into
  a complete OGS model. Can be created from components or loaded from disk.
  - `Model.run()` executes the simulation and returns a `Simulation` object
  - `Model.plot_constraints()` visualizes boundary conditions and source terms
  - Supports all storage operations (save, load by folder or ID)
- **Simulation**: Represents a completed (ongoing by SimulationController) OGS simulation with model and results.
  - `Simulation.meshseries` provides access to the simulation output
  - `Simulation.log` provides access to parsed log file as a `Log` object
  - `Simulation.status` and `Simulation.status_str` track simulation state
  - Full save/load support with automatic storage of model, results, and logs
- **Result**: Wrapper for simulation results providing convenient access to mesh series.
  - Direct indexing to access timesteps (e.g., `result[-1]` for final timestep)
  - Integrates with existing MeshSeries functionality
- **Execution**: Manages OGS execution parameters (parallelization, logging, etc.)
  - Configurable OMP threads and assembly threads
  - Execution from YAML files or programmatic configuration
- **Log**: Parser and analyzer for OGS log files
  - `convergence_newton_iteration()` and `convergence_coupling_iteration()` extract
    convergence data
  - `plot_convergence()` visualizes convergence behavior
  - `simulation_info()` and `termination_info()` provide simulation metadata
- **Storage System** (`SaveBase`): Unified base class for all saveable objects
  - Configurable user path via `SaveBase.Userpath` and backup strategy via `SaveBase.Backup` flag
  - ID-based organization in class-specific subdirectories
  - Archive mode for creating portable, self-contained copies
  - Temporary storage support for intermediate objects

### mesh

- added `ot.mesh.node_reordering` to fix node ordering of a mesh to comply with OGS standards
- added `ot.mesh.validate` to check, whether a mesh complies with OGS standards

### gmsh_tools

- added more control over the discretization for `remesh_with_triangles`.

### MeshSeries

- Now has a difference method.
- added support for logarithmic scaling in `ot.MeshSeries.plot_time_slice`.
- new functions for renaming domain and subdomains
- can now be scaled spatially / temporally in the Constructor
- added `interpolate` method, to interpolate the MeshSeries data on a new mesh
- added `compare` method to `ot.MeshSeries`.
- Storage support: can be saved and loaded by folder or ID

### Meshes

- save function performs partmesh if number of partitions are given, with optional dry_run
  - create_partitioning() and create_metis if vtu files are already present
- subdomains() -> subdomain, domain_name() -> domain_name (with setter)
- `plot` method displays domain mesh and subdomains (2D domain only for now)
- `remove_material` method removes specified material id from domain and
  updates subdomains accordingly
- `modify_names` method extends mesh names with prefix and/or suffix
- `Meshes.from_mesh` now also works for 3D domain meshes
- now checks datatypes of MaterialIDs, bulk_node_ids, bulk_element_ids and coordinates upon saving
- `from_file()` method to restore from meta.yaml
- Storage support with metadata preservation

### Project

- dependencies() return a list of referenced/needed files (meshes, xml includes, python scripts)
- plot_constraints() -> overview plot with boundary conditions and source terms
- Enhanced `__repr__()` and `__str__()` for better object inspection
- Storage support with proper XML handling

### plot

- add option to plot legend outside of plots.
- added support for continuous contourplots via `continuous_cmap=True`.
- Allow user to set arrowsize parameter in contourf

### Variable

- added aggregation methods `min`, `max`, `mean`, `median`, `sum`, `std`, `var`.

## Infrastructure

- Gallery example figures are now tested against hashes which avoids
  unexpectedly changing figures.

## Documentation

### New Examples

- **plot_framework_short.py**: Condensed workflow demonstrating the complete OGSTools
  framework (Setup → Compose → Run → Analyze → Store)
- **plot_storage.py**: Comprehensive guide to the storage system covering basic usage,
  ID-based organization, overwriting with backup, archiving, and advanced topics

### Updated Examples

- Updated simulation examples to use new `Model` and `Simulation` classes
- Modified examples to demonstrate the new storage system capabilities
- Added example data: small_deformation simulation with complete model, execution,
  meshes, and results

### Tests

- plotting tests now actually check whether the resulting figure is as expected
- Added comprehensive tests for new framework components:
  - `test_framework.py`: Tests for Model, Simulation, Result integration
  - `test_model.py`: Model class functionality
  - `test_simulation.py`: Simulation lifecycle and storage
  - `test_storage.py`: Storage system, backup, and archive functionality

### Imports

- Top-level API now exports additionally: `Model`, `Simulation`, `SimulationController`, `Execution`, `Log`

## Maintainer TODOs

### next sub release

### next main release
