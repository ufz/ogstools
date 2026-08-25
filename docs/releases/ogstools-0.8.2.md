# OGSTools 0.x Release Notes

Python 3.12-3.14
OGS 6.5.8 -

# Breaking changes

## API breaking changes

## Deprecations

### Removed Modules

### Examples

# Changes (non API-breaking)

## Bugfixes

- fixed sorting in Meshes.from_files (if key was not in the beginning, it didn't sort correctly)
- fixed broken weblinks in documentation
- allow values for group based parameters in `Project.parameters` (allows to set vectorial or anisotropic group values. Previously, only allowed scalars)
- fixed `Project.process_variables.add_bc` and `Project.process_variables.add_st` to allow all different kinds of types. Previously, some
  checks caused unintentional errors.
- fixed `Meshseries.probe` with linear interpolation and cell_data yielding nan at the mesh boundary as cell centers are used for interpolation. Now used nearest neighbor as a fallback.
- fixed reading xdmf data where some data field are only present in the first timestep. Now they will also be read for later timesteps.

## Features

- plot
  - added `xlim` and `ylim` as arguments for `contourf`
  - improved `meshes.plot`
  - fixed line ordering by trying to sort by cell adjacency
  - `contourf`: corner elements were missing if none of their points were in the selected x- or y-limits; now they are checked explicitly.
- meshes
  - `Meshes.from_mesh` supports quadratic mesh
- mesh
  - `filepath` is now attached to mesh upon reading (with `ot.mesh.read`)
- Project
  - `BuildTree.populate_tree` now can handle dicts as the element text
    (in this case the key and value pairs will create corresponding subelements)
  - Added `added deactivate_subdomain` to `Project.process_variables`
- variables
  - dataname of a variable is now also looked for in the attributes of mesh (e.g. points) or meshseries (e.g. timevalues)
  - functions of nested variables are now stored in list of Functions (instead of nested lambdas) which allows to retrospectively modify arguments and parameters
  - arguments of stored functions can be strings which represent the names of data inside a mesh or meshseries to allow for variables depending on different input data
  - outputname can now explicitly be an empty string

## Infrastructure

- added lychee make command to check for broken links in the docs

## Documentation

### New Examples

- new meshes example (Selke)

### Updated Examples

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release
