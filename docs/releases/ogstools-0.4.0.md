# OGSTools 0.4.0 Release Notes

## Overview

Supports Python: 3.10, 3.11, 3.12.

First release after integration of ogs6py. The state of ogs6py corresponds to the functionality of  [v.403 of ogs6py](https://github.com/joergbuchwald/ogs6py/releases/tag/v.403) (only light API changes, see Features/Project file)

## OGS

[OpenGeoSys](https://www.opengeosys.org/) version: 6.5.3

## Migration strategy from ogs6py to ogstools

The recommended strategy is:

- Back up your environment with `pip freeze > yourfile.txt`
- Upgrade ogs6py to 0.403 first and resolve issues (see breaking changes).
- Uninstall ogs6py and install ogstools
- Remove `import ogs6py`. Add `import ogstools as ot`. The former `OGS` becomes `Project` and function parameters of `OGS.__init__()` are now with lower case names. See example in Features

## API changes

### plot

- renamed meshplotlib to plot
- renamed plot function to contourf
- renamed setup.p_min -> setup.vmin, setup.p_max -> setup.vmax
- added several setup options available as kwargs

### variables

- renamed propertylib to variables (to align with OpenGeoSys vocabulary)
- presets are now directly accessible as variables
- renamed presets to properties
- renamed Property class to Variable

## meshlib

- renamed aggregate function to time_aggregate
- renamed MeshSeries.read to MeshSeries.mesh

## Features

### ogs6py

- new version with significant parts of the codebase rewritten. Changes visible to the user:

#### example

before:

```python
import ogstools as ot

prj = ot.Project(input_file="mechanics.prj", output_file="old_parameter_add.prj")
prj.add_block(
    blocktag="parameter",
    parent_xpath="./parameters",
    taglist=["name", "type", "value"],
    textlist=["density", "Constant", "1900"],
)
prj.write_input()
```

now:

```python
import ogstools as ot

prj = ot.Project(input_file="mechanics.prj", output_file="new_parameter_add.prj")
prj.parameters.add_parameter(name="density", type="Constant", value="1900")
prj.write_input()
```

- support for combining replacement- and build-method commands
- breaking changes: some methods renamed closer to project file tags:

* `add_process_variable()` split-up into `add_secondary_variable()` and  `add_process_variable()`
* `geo.add_geom()` -> `geometry.add_geometry()`, `timeloop` -> `time_loop` etc.

- support for more MPL properties
- several bugfixes and tiny new features
- ## integration into ogstools

### plot

- Time slice plots can be created easily: filled contour plots of transient data along a sampling line ([example](https://ogstools.opengeosys.org/auto_examples/howto_plot/plot_timeslice.html#))

![time_slice](https://ogstools.opengeosys.org/_images/sphx_glr_plot_timeslice_001.png)

- Variables now have default symbols (e.g. used for labels in plots)

### Mesh and MeshSeries

- Mesh can be created from a shapefile

```
mesh = ot.Mesh.read(test_shapefile)
```

- Mesh can be remeshed with triangle
  ![Mesh_triangle](https://ogstools.opengeosys.org/_images/sphx_glr_plot_remeshing_002.png)
- MeshSeries has a performant algorithm for integration point tessellation called `MeshSeries.ip_tesselated()` -  [example for analyzing integration point data](https://ogstools.opengeosys.org/auto_examples/howto_postprocessing/plot_ipdata.html#sphx-glr-auto-examples-howto-postprocessing-plot-ipdata-py)
- MeshSeries allows multidimensional indexing on ndarrays <https://numpy.org/doc/stable/user/basics.indexing.html>

```python
import ogstools as ot

ms = ot.MeshSeries("filepath/filename_pvd_or_xdmf")
ms.data("darcy_velocity")[-2:, 1:4, :]  # shape is(2, 3, 2)
result_mesh = ms[-1]
for mesh in ms:
    print(mesh)
```

- Added function argument to Meshseries.animate functions has more flexible parameter that allow
  1. transformation of the mesh and
  1. customization of the plot
- Meshseries has domain_aggregate function (e.g. min/max of a variable per time step)

### Project files

- ogs6py added (this version corresponds to https://github.com/joergbuchwald/ogs6py/releases/tag/v.403)
- OGS class is renamed to Project
- Function parameters are with now lower case letters
- `project_file` is now `output_file`

### Documentation, examples and more

- Documentation got new structure of examples (now all organized under ogstools/examples/)
- pip requirements.txt files (of tested environment) are added for stable test environment reproduction (To be used by developers. Do not use it in your projects!)

## Bugfixes

- Several small fixes in plotting functions (visual)
- MeshSeries closes file handle to h5 file after reading is finished
- Dependency compatibility (e.g., remove restriction to matplotlib and relaxing requirements)
