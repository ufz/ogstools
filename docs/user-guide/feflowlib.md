# feflowlib - feflow2ogs

```{eval-rst}
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

## Introduction

The converter is used to convert data stored in FEFLOW binary format to VTK format (`.vtu`).
This converter was developed in the Python language and interacts with the Python API of FEFLOW.
It allows the use of `pyvista` especially for the creation of unstructured grids.
But it can also be used as a library to easily access FEFLOW data in Python.
With the usage of [`ogs6py`](https://joergbuchwald.github.io/ogs6py-doc/index.html) it is possible to create a `prj-file` from the converted model to enable simulations with `OGS`.
At the moment only `steady state diffusion` and `liquid flow` processes are supported to set up the `prj-file`.

## Features

All in all, the converter can be used to convert steady state diffusion and liquid flow processes from FEFLOW.
This includes the conversion of the bulk mesh together with the boundary conditions, as well as the creation of the corresponding mesh `vtk-files`.
In addition, (in)complete `prj files` can be created automatically.
The `prj file` is set up of a model-specific part and a part that is read from a template and defines the solver and process configuration.
The current status enables:

### Main features:

- conversion of FEFLOW meshes
- extraction of boundary condition
- creation of OGS-models for `steady state diffusion` and `liquid flow` processes
- usage via *command line interface* or as *Python library*

### specific features:

- get point, cells and celltypes to array according to pyvista convention for [pyvista.UnstructuredGrid](https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.html)
- write MaterialIDs to a dictionary
- write point and cell data with MaterialIDs to dictionaries that match the points and cells of the input data
- convert only the geometry of input data
- update the geometry with point and cell data
- convert the geometry with point and cell data
- extraction and writing of material specific meshes that represent inhomogeneous material properties
- prepare FEFLOW data for ogs simulation with tools that allow:
  - creation of (in)complete `prj-files` for `OGS`
    - model specific elements refer to mesh, material properties, parameter, boundary conditions
    - templates define the solver, time loop, process, output
  - writing of boundary conditions to separate `.vtu`-files

## Requirements

- Running via [Docker](./docker.md)

**OR**:

- FEFLOW
- ogstools with feflow (`pip install ogstools[feflow]`)

## Installation

The converter requires FEFLOW to be installed.
There are different ways to fulfill this requirement, either one installs FEFLOW or works with a container that has FEFLOW installed.

::::{tab-set}
:::{tab-item} Linux
:sync: unix
Depending on the Linux distribution used, different steps are possible to install FEFLOW.
The DHI supports the installation of FEFLOW on Ubuntu.
Instructions for installing from an apt repository can be found on their [website](https://download.feflow.com/download/FEFLOW/linux/).
Additionally, the environment variables must be set correctly to find the FEFLOW installation in Python.
The following three variables need to be set:

```bash
export PYTHONPATH=/opt/feflow/8.0/python
export LD_LIBRARY_PATH=/opt/feflow/8.0/lib64:/opt/feflow/common/qt/lib64:/opt/feflow/common/lib64
export FEFLOW80_ROOT=/opt/feflow/8.0
```

:::
:::{tab-item} macOS
:sync: unix
DHI, the developer of FEFLOW, do not support macOS.
So, it is needed to use the [Docker container](./docker.md) or a virtual machine.
:::
:::{tab-item} Windows
:sync: win
FEFLOW is fully supported on Windows.
The installation is according to the official website.
:::
::::

The installation of the FEFLOW converter with `ogstools` is optional.
The option can be selected with the following `pip`-command.

```
pip install ogstools[feflow]
```

## Command line usage

`feflow2ogs` is a command line interface of the converter that summarizes the main functions to provide the user with an accessible application.

```{argparse}
---
module: ogstools.feflowlib._cli
func: parser
prog: feflow2ogs
---
```

## API usage

In addition, it may be used as Python module.
Further information can be found at: [](../reference/ogstools.feflowlib).

## Example

An example of how the API can be used is given at: [](../auto_examples/howto_feflowlib/index).
