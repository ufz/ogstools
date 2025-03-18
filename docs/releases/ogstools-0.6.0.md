# OGSTools 0.6.0 Release Notes (upcoming release)

OGS version: 6.5.4
Python: 3.10 - 3.13

## API breaking changes

- meshseries.probe now squeezes the returned array: it seems more intuitive
  to return a 1D array if no list of points is provided (just a single tuple)

- meshseries.plot_time_slice has a new user interface - see the update example
  or the API reference. The interpolate argument was removed, as it tempts you to
  be used as a default (as it produces a nice image), but in doing so, can easily
  lead to wrong conclusions as the interpolated image might differ significantly
  from the raw data.

- generalized meshseries.animate: it is now a free standing function
  (plot.animate) and can take any plotting function - see the updated example or
  the API reference

## Features

- MeshSeries can now be initialized with `from_data(meshes, timevalues)`
- MeshSeries now has an .items() iterator
- MeshSeries has now an .extend function to combine 2 MeshSeries (e.g. for simulation restart/continuation)
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
- OGS_BIN_PATH is read, this allows to specify a location of a custom OGS (ogs wheel not needed)
- Improved errors and warnings when system has multiple or no OGS on PATH
- ogstools.status(verbose=True) prints status of OGS installation

## Infrastructure

### Tests

- Tests can be marked as system and tools
  - system tests: Invoke running a simulation with OGS
  - tools tests: Invoke calls to OGS binary tools and pybind11 functionality (in future)

### Imports

- When imports are actually implementation details they should be loaded within the calling function (delayed initialization). Applied to imports that rely on OGS installation.
