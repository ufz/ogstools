# OGSTools 0.x Release Notes (upcoming release)

This is not released yet!

## API breaking changes

- MeshSeries.data --> MeshSeries.values
- MeshSeries.clear --> MeshSeries.clear_cache
- meshlib.gmsh_meshing.remesh_with_triangle --> meshlib.gmsh_meshing.remesh_with_triangles
- msh2vtu python interface was replaced with meshes_from_gmsh
  - CLI tool msh2vtu is not affected by this
  - parameter keep_ids was removed (in our OGS world there is no reason to keep the gmsh data names and the wrong data types in the meshes, which would happen if k was used)
  - parameter log_level was changed to log (True or False)

## Features

- MeshSeries gets copy() method.
- MeshSeries gets transform() method, that applies an arbitrary transformation function to all time steps.
- MeshSeries get extract() method to select points or cells via ids
- MeshSeries can be sliced to get new MeshSeries with the selected subset of timesteps
- difference() between two meshes is now possible even with different topologies

## Infrastructure

- Python 3.13 support (CI testing)
- Testing of all supported Python version 3.10-3.13 (pip and conda)
- Testing with pinned dependencies in regression tests and with open dependencies in maintenance tests
