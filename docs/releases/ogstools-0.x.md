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

## Features

- plot
  - added `xlim` and `ylim` as arguments for `contourf`
  - improved `meshes.plot`
  - fixed line ordering by trying to sort by cell adjacency
- meshes
  - `Meshes.from_mesh` supports quadratic mesh
- mesh
  - `filepath` is now attached to mesh upon reading (with `ot.mesh.read`)
- Project
  - `BuildTree.populate_tree` now can handle dicts as the element text
    (in this case the key and value pairs will create corresponding subelements)
  - Added `added deactivate_subdomain` to `Project.process_variables`

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
