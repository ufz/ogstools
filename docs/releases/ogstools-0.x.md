# OGSTools 0.x Release Notes (upcoming release)

This is not released yet!

## API breaking changes

- meshseries.probe now squeezes the returned array: it seems more intuitive
  to return a 1D array if no list of points is provided (just a single tuple)

- meshseries.plot_time_slice has a new user interface - see the update example
  or the API reference. The interpolate argument was removed, as it tempts you to
  be used as a default (as it produces a nice image), but in doing so, can easily
  lead to wrong conclusions as the interpolated image might differ significantly
  from the raw data.

- generalized meshseries.animate: its now a free standing function
  (plot.animate) and can take any plotting function - see the updated example or
  the API reference

## Bugfixes

## Features

- MeshSeries can now be initialized with `from_data(meshes, timevalues)`
- MeshSeries now has an .items() iterator
- plot.line now automatically sorts the data
- User can select format when saving animation (gif or mp4)
- variables.vector add BHE vector with support for slicing its components
- Variables now have methods to create Variables related to absolute error, relative error and analytical solution corresponding to the parent Variable
- Variable.transform now also works on MeshSeries
- plot.line now can also handle MeshSeries - this allows to draw an array of lines in one call (e.g. lines for each pt in the MeshSeries over time or lines for each timestep over a spatial coordinate)
- plot.contourf now only returns a figure if it wasn't given one as an argument
- improved axis labelling for shared axes (only the outer axes should get labelled now)
- new: meshseries.resample to interpolate the meshseries to new timevalues
- new: extract_probe to create a new meshseries consisting only of points and
  probed data at these points

## Infrastructure

## Maintainer TODOs

### next sub release

### next main release

- MeshSeries: from_data() constructor -> __init__()
