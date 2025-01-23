# OGSTools 0.5.0 Release Notes

- Recommended OGS Version: 6.5.4

## API breaking changes

- MeshSeries.data --> MeshSeries.values
- MeshSeries.clear --> MeshSeries.clear_cache
- In aggregate functions func str is replaced by callables (e.g. numpy.min)
- meshlib.gmsh_meshing.remesh_with_triangle --> meshlib.gmsh_meshing.remesh_with_triangles
- msh2vtu python interface was replaced with meshes_from_gmsh
  - CLI tool msh2vtu is not affected by this
  - parameter keep_ids was removed (in our OGS world there is no reason to keep the gmsh data names and the wrong data types in the meshes, which would happen if k was used)
  - parameter log_level was changed to log (True or False)
- removed:
  - MeshSeries.spatial_data_unit/spatial_output_unit/time_unit (see
    MeshSeries.scale())
  - plot.linesample/linesample_contourf
  - meshlib.data_processing.interp_points/distance_in_profile/sample_polyline
    (see updated line sample example)

## Bugfixes

- Failed sub library imports led to incomplete and unhandled package import
- MeshSeries was unable to handle xdmf with just one timestep correctly
- MeshSeries kept the hdf5 file handle open - parallel read access was not possible
- OMP_NUM_THREADS was not working on Windows
- Feflow mesh did not allow mixed celltypes
- plot functions had sometimes different color schemes in the color bar
- Tortuosity was not a medium property
- BHE mesh (z coordinate negative)

## Features

- MeshSeries gets copy() method.
- MeshSeries gets transform() method, that applies an arbitrary transformation function to all time steps.
- MeshSeries get extract() method to select points or cells via ids
- MeshSeries can be sliced to get new MeshSeries with the selected subset of timesteps
- MeshSeries gets a modify function that applies arbitrary function to all timestep - meshes.
- MeshSeries gets a save function (only for pvd implemented yet)
- difference() between two meshes is now possible even with different topologies
- Project write_input, path can be specified
- MeshSeries gets scale() method to scale spatially or temporally
- variables.get_preset will now return a Variable corresponding to the spatial
  coordinates if given "x", "y" or "z"
- plot module gets line() function as a general purpose 1D plotting function
- plot.setup get spatial_unit and time_unit which are used for labeling

## Infrastructure

- Python 3.13 support (CI testing)
- Testing of all supported Python version 3.10-3.13 (pip and conda)
- Testing with pinned dependencies in regression tests and with open dependencies in maintenance tests
- msh2vtu - complete overhaul

## Examples

- All examples use `import ogstools as ot`. To not be confused with ogs python bindings

# Footnotes

- All related Merge requests are tagged with 0.5.0 Release https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests?scope=all&state=merged&milestone_title=0.5.0%20Release
