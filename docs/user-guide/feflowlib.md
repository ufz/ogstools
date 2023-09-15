# FEFLOW converter - feflow2ogs

```{eval-rst}
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

## Introduction

The converter is used to convert data stored in FEFLOW binary format to VTK format (`.vtu`).
This converter was developed in the Python language and interacts with the Python API of FEFLOW.
It allows the use of `pyvista` especially for the creation of unstructured grids.
But it can also be used as a library to easily access FEFLOW data in Python.

## Features

- write point, cells and celltypes to array according to pyvista convention
- write MaterialIDs to a dictionary
- write point and cell data with MaterialIDs to dictionaries that match the points and cells of the input data
- read only the geometry of input data
- update the geometry with point and cell data
- read the geometry with point and cell data
- prepare FEFLOW data for ogs simulation with tools that allow:
  - writing of xml-snippets
  - writing of boundary conditions to separate `.vtu`-files

## Requirements

- Running via [Docker](./docker.md)

**OR**:

- FEFLOW
- ogstools with feflow (`pip install ogstools[feflow]`)
- [ifm_contrib](https://github.com/red5alex/ifm_contrib), a library for extending the functionalities of the FEFLOW Python API.

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
The option can be be selected with the following `pip`-command.

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
