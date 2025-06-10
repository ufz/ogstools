# OGSTools 0.7.0 Release Notes

OGS version: 6.5.4
Python: 3.10 - 3.13

# Breaking changes

- OGS Wheel (`pip install ogs`) is now an optional (before mandatory) dependency of OGSTools. OGSTools requests either a OGS wheel OR a custom OGS made available on PATH or OGS_BIN_PATH.

## API breaking changes

- mesh.read_shape (removed function, functionality is not covered anymore)

## Deprecations

- meshseries.plot_probe (instead use MeshSeries.extract_probe and plot.line)

### Examples

- removed all examples with shape files

# Changes (non API-breaking)

## Bugfixes

- meshes_from_gmsh (msh2vtu): Physical groups, which include other physical groups needed to be defined after the subgroups otherwise groups were comprised of the wrong elements, now it's fixed
- MeshSeries: probe() failed with output meshes only consisting of multiple lines representing BHEs, this is for example the case by using the output option by material id
- MeshSeries: indexing failed with numpy data types
- Feflow converter: Heterogeneous material for properties fixed (for KF only)
- Feflow converter: In CLI extraction of topsurface domain fixed
- BHE: Fix for huge BHE arrays
- plot: Fix for plots from parallel computation (vtkGhostType)

## Features

- Logparser: analysis to inspect simulation behaviour over clock time and model time
- Logparser: plot to create an overview of the convergence behavior for the
  entire simulation (ot.logparser.plot_error, ot.logparser.plot_convergence_order)
- Logparser: Functionality for real time monitoring
- Logparser: Can consume new version (2) of OGS log files (can still consume version 1)
- OGS simulation can be run in a background process via ot.Project.run_model(..., background=True)
- MeshSeries.extract_probe/probe/values accept str/Variables and lists of them as arguments (with improved performance)
- plot: handle gaps in sampling lines and disjoined line meshes, such that each individual region is drawn separately.
- Project: Run simulations in background
- Project: Showcasing a second variant of setting well defined initial pressures in the gallery

## Infrastructure

- Several fixes for pyvista>=0.45
- Pagefind index generation

### Tests

- Hypothesis testing introduction
- More parallel and parameterized tests

### Usability

- Clarify what is expected from the user-provided list of observation points in plot_probe.
- Some improved error messages

## Maintainer TODOs

### next sub release

### next main release

- MeshSeries: from_data() constructor -> __init__()
