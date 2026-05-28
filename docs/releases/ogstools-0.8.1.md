# OGSTools 0.8.1 Release Notes

Python 3.12-3.14
OGS 6.5.8 -

# Breaking changes

## API breaking changes

## Deprecations

### Removed Modules

### Examples

# Changes (non API-breaking)

## Bugfixes

- MeshSeries: could not be read if xdmf-file was a symbolic link
- added missing tools to cli interface (`mergeMeshToBulkMesh`, `CreateAnchors`)
- identifysubdomains will ignore anchor meshes to not wrongly overwrite the special id mapping
- fixed sorting of `ot.plot.line` in some edge cases
- A Matrix Variable transformed to the polar coordinate system
  (`ot.variables.Matrix.to_polar()`) is now also able to use `.transform` on a MeshSeries
- `.xdmf` MeshSeries which only contain points (and no cells) can now be properly
  read
- Project: Curves (bin) and chemical solver database (dat) are now copied and saved with Project
- Materiallib: fix bug with selecting multiple materials in a copy
- ip_mesh: fix for meshes with mixed cell types
- `plot.contourf`: fix when used in notebooks
- `MeshSeries.from_data`: fix save and scale behavior
- OGS-Restart: minor fixes

## Features

- added the Class `ogstools.mesh.IPdata` to interface with the integration point
  metadata
- variable components can now be plotted via strings e.g. `"displacement_x"` will
  plot the same thing as `ot.variables.displacement["x"]`
- added standalone type hinted methods, to set parameters to a project per
  parameter type e.g. `prj.parameters.set_constant_parameter` / `prj.parameters[name] = int_or_float` or
  `prj.parameters.set_function_parameter` / `prj.parameters[name] = str`
- added `prj.parameters.set_group_parameter`
- `ogsmonitor` is now a standalone CLI tool (previously only usable from Jupyter
  notebooks). New CLI features include automatic port selection, pipe support, better
  error codes, and the logfile path is now a positional argument.
- temporary files are now all consistently written und `/tmp/ogstools_<username>/<functionname>/`

## Infrastructure

## Documentation

### New Examples

- added an example which shows, how to modify integration point data

### Updated Examples

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release

- modify ip_data_threshold to use the same pattern as in the ip_data.py example
- MaterialManager: set_material -> set_material_from_file + set_material(self, Material)
