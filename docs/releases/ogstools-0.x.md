# OGSTools 0.x Release Notes (upcoming release)

This is not released yet!

## API breaking changes

- meshseries.probe now squeezes the returned array: it seems more intuitive
  to return a 1D array if no list of points is provided (just a single tuple)

## Bugfixes

## Features

- MeshSeries can now be initialized with `from_data(meshes, timevalues)`
- MeshSeries now has an .items() iterator
- plot.line now automatically sorts the data
- User can select format when saving animation (gif or mp4)

## Infrastructure

## Maintainer TODOs

### next sub release

### next main release

- MeshSeries: from_data() constructor -> __init__()
