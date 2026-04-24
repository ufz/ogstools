# OGSTools 0.x Release Notes

Python 3.11-3.14
OGS 6.5.7 -

# Breaking changes

## API breaking changes

## Deprecations

### Removed Modules

### Examples

# Changes (non API-breaking)

## Bugfixes

## Features

- added the Class `ogstools.mesh.IPdata` to interface with the integration point
  metadata
- variable components can now be plotted via strings e.g. `"displacement_x"` will
  plot the same thing as `ot.variables.displacement["x"]`
- added standalone type hinted methods, to set parameters to a project per
  parameter type e.g. `prj.parameters.set_constant_parameter` / `prj.parameters[name] = int_or_float` or
  `prj.parameters.set_function_parameter` / `prj.parameters[name] = str`
- added `prj.parameters.set_group_parameter`

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
