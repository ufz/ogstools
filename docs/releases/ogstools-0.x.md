# OGSTools 0.7.0 Release Notes

# Breaking changes

## API breaking changes

## Deprecations

### Examples

# Changes (non API-breaking)

## Bugfixes

## Features

- MeshSeries
  - has now has difference method.
  - added support for logarithmic scaling in `ot.MeshSeries.plot_time_slice`.
  - new functions for renaming domain and subdomains
- Meshes
  - save function performs partmesh if number of partitions are given
  - subdomains() -> subdomain, domain_name() -> domain_name (with setter)
- plot
  - add option to plot legend outside of plots.
  - added support for continuous contourplots via `continuous_cmap=True`.
  - Allow user to set arrowsize parameter in contourf

## Infrastructure

- Gallery example figures are now tested against hashes which avoids
  unexpectedly changing figures.

## Documentation

### Tests

- plotting tests now actually check whether the resulting figure is as expected

### Imports

## Maintainer TODOs

### next sub release

### next main release
