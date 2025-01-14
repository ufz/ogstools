# OGSTools 0.x Release Notes (upcoming release)

This is not released yet!

## API breaking changes

- MeshSeries.data --> MeshSeries.values
- MeshSeries.clear --> MeshSeries.clear_cache
- meshlib.gmsh_meshing.remesh_with_triangle --> meshlib.gmsh_meshing.remesh_with_triangles

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
