# OGSTools 0.5.0 Release Notes

- Recommended OGS Version: 6.5.4

## API breaking changes

- FeflowModel-class introduced:
  - FeflowModel.mesh -> to get the mesh with all FEFLOW properties, replaces feflowlib.convert_properties_mesh()
  - FeflowModel.ogs_bulk_mesh -> to get the mesh with only materialIDs
  - FeflowModel.subdomains -> access the boundary conditions and the source terms of the FEFLOW model, replaces feflowlib.extract_point_boundary_conditions(), feflowlib.extract_cell_boundary_conditions()
  - FeflowModel.process -> see the process to be simulated
  - FeflowModel.material_properties -> access the material_properties, replaces feflowlib.get_material_properties_of_H_model(), feflowlib.get_material_properties_of_HT_model(), feflowlib.get_material_properties_of_CT_model()
  - FeflowModel.project -> access the project file
  - FeflowModel.setup_prj() -> create the project file, replaces feflowlib.setup_prj_file()
  - FeflowModel.save() -> save the mesh, subdomain, and the project-file
  - FeflowModel.run() -> run the FEFLOW model in OGS
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

- FeflowModel: fix material_properties for HT process
- FeflowModel: could not convert mixed celltypes in FEFLOW mesh
- Failed sub library imports led to incomplete and unhandled package import
- MeshSeries was unable to handle xdmf with just one timestep correctly
- MeshSeries kept the hdf5 file handle open - parallel read access was not possible
- OMP_NUM_THREADS was not working on Windows
- plot functions had sometimes different color schemes in the color bar
- Tortuosity was not a medium property
- BHE mesh (z coordinate negative)

## Features

- FeflowModel: heterogeneous material properties are now saved on the mesh and not a separate mesh
- FeflowModel: allow generic creation of project files for unsupported processes to have a proposal of a project file, which needs to be modified manually to have working OGS model
- FeflowModel: allow configuration of time stepping and error tolerance for all processes
- FeflowModel: use materialIDs from FEFLOW-user-data, if defined
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
- New examples to show post-conversion modifications of FEFLOW model(modify boundary conditions and project-file)

# Footnotes

- All related Merge requests are tagged with 0.5.0 Release https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests?scope=all&state=merged&milestone_title=0.5.0%20Release
