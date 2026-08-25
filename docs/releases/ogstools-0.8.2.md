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

- fixed sorting in Meshes.from_files (if key was not in the beginning, it didn't sort correctly) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- fixed broken weblinks in documentation ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- allow values for group based parameters in `Project.parameters` (allows to set vectorial or anisotropic group values. Previously, only allowed scalars) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- fixed `Project.process_variables.add_bc` and `Project.process_variables.add_st` to allow all different kinds of types. Previously, some
  checks caused unintentional errors. ([!471](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/471))
- fixed `Meshseries.probe` with linear interpolation and cell_data yielding nan at the mesh boundary as cell centers are used for interpolation. Now used nearest neighbor as a fallback. ([!473](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/473))
- fixed reading xdmf data where some data field are only present in the first timestep. Now they will also be read for later timesteps. ([!473](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/473))

## Features

- plot
  - added `xlim` and `ylim` as arguments for `contourf` ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - improved `meshes.plot` ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - fixed line ordering by trying to sort by cell adjacency ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - `contourf`: corner elements were missing if none of their points were in the selected x- or y-limits; now they are checked explicitly. ([!482](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/482))
- meshes
  - `Meshes.from_mesh` supports quadratic mesh ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- mesh
  - `filepath` is now attached to mesh upon reading (with `ot.mesh.read`) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- Project
  - `BuildTree.populate_tree` now can handle dicts as the element text
    (in this case the key and value pairs will create corresponding subelements) ([!472](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/472))
  - Added `added deactivate_subdomain` to `Project.process_variables` ([!472](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/472))
- variables
  - dataname of a variable is now also looked for in the attributes of mesh (e.g. points) or meshseries (e.g. timevalues) ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))
  - functions of nested variables are now stored in list of Functions (instead of nested lambdas) which allows to retrospectively modify arguments and parameters ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))
  - arguments of stored functions can be strings which represent the names of data inside a mesh or meshseries to allow for variables depending on different input data ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))
  - outputname can now explicitly be an empty string ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))

## Infrastructure

- added lychee make command to check for broken links in the docs ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))

## Documentation

### New Examples

- new meshes example (Selke) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))

### Updated Examples

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release
