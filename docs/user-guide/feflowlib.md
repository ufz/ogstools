# feflowlib - feflow2ogs

```{eval-rst}
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

## Introduction

`feflowlib` is a Python module to access **FEFLOW** data and prepare it for **OpenGeoSys (OGS)**.
`feflow2ogs` is the associated *command line interface* that combines the most important `feflowlib` functions in a user-friendly workflow.
Together they are referred to as FEFLOW-converter, as they enable the conversion of data stored in FEFLOW binary format to VTK format (`.vtu`) and the preparation of OGS-models.-processes
The converter interacts with the Python API (`ifm`) of the proprietary software FEFLOW.
`pyvista` is used especially for creating unstructured grids/meshes.
With the usage of [`ogs6py`](./ogs6py.md) it is possible to create a proposal of a `prj-file` from the converted model to enable simulations with OGS.
At the moment `steady state diffusion`, `liquid flow`, `hydro thermal` and `component/mass transport` processes are supported to set up complete `prj-files`.
For other processes, a generic `prj-file` is created that needs manual configurations to be ready for OGS simulation.

## Features

All in all, the converter can be used to convert automatically and OGS-ready models for the following processes: H (`steady state diffusion`, `liquid flow`), HT (`hydro thermal`) and HC `component transport` models from FEFLOW.
This includes converting the bulk mesh together with the boundary conditions and source terms and creating the corresponding mesh `vtk-files`.
In addition, (in)complete `prj-files` can be created automatically.
The `prj-file` is composed of a model-specific part and a part that is read from a template that defines the solver and time configuration.
This means that the converter supplies a suggestion for a `prj-file` that is not guaranteed to work.
The current status enables:

### Main features

- Conversion of FEFLOW meshes
- Extraction of boundary condition
- Automatic creation of OGS-models for `steady state diffusion`, `liquid flow`, `hydro thermal` and `component transport` processes
- Automatic creation of `prj files` for all types of processes to speed up OGS model setup
- Usage via *command line interface* or as *Python library*

### specific features

- Get points, cells and celltypes to array according to pyvista convention for [pyvista.UnstructuredGrid](https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.UnstructuredGrid.html)
- Write Material properties in a dictionary
- Convert only the geometry of input data
- Update the geometry with point and cell data
- Convert the geometry with point and cell data
- Prepare FEFLOW data for OGS simulation with tools that allow:
  - creation of (in)complete `prj-files` for OGS
    - model specific elements refer to mesh, material properties, parameters, boundary conditions
    - templates define the solver, time loop, process, output
  - writing of boundary conditions to separate `.vtu`-files

## Data flow chart

The following diagram shows the data flow that occurs in the `feflowlib`, from a FEFLOW model to an OGS simulation.
`feflow2ogs` summarizes all the necessary features from the `feflowlib` to enable this data flow.

```{mermaid}
graph TD
 FEFLOW(FEFLOW model):::FEFLOWStyle -->|feflowlib| OGS_PRJ:::InputStyle
 FEFLOW(FEFLOW model):::FEFLOWStyle -->|feflowlib| OGS_BULK:::InputStyle
 FEFLOW(FEFLOW model):::FEFLOWStyle -->|feflowlib| OGS_BOUNDARY:::InputStyle
 FEFLOW(FEFLOW model):::FEFLOWStyle -->|feflowlib| OGS_SOURCE:::InputStyle
 SSD(steady state diffusion <br> liquid flow <br> hydro thermal <br> component transport):::TemplateStyle -->|template| OGS_PRJ:::InputStyle
 OGS_PRJ[project file]:::InputStyle -->|xml format| OGS
 OGS_BULK[bulk mesh]:::InputStyle -->|vtu format| OGS
 OGS_BOUNDARY[boundary meshes]:::InputStyle -->|vtu format| OGS
 OGS_SOURCE[source term meshes]:::InputStyle -->|vtu format| OGS
 OGS(OpenGeoSys):::OGSStyle -->|vtu format| OGS_PRESSURE[simulation results]:::OGSOutputStyle


classDef InputStyle fill:#9090ff
classDef OGSStyle fill:#104eb2, color:#ffffff
classDef FEFLOWStyle fill:#1e690a, color:#ffffff
classDef feflowlibStyle fill:#081f6a, color:#ffffff
classDef OGSOutputStyle fill:#a0a0f0
classDef TemplateStyle fill:#009c21, color:#ffffff
```

## Requirements

- Running via [Docker](./docker.md)

**OR**:

- FEFLOW
- ogstools and a FEFLOW installation (`pip install ogstools[feflow]`)

## Installation

The converter requires FEFLOW to be installed.
There are different ways to fulfil this requirement, either one installs FEFLOW or works with a [Docker container](./docker.md) that has FEFLOW installed.

::::{note}
If you use an old version of FEFLOW, make sure it is compatible with `Python 3.10`, as this is a requirement for using ogstools and the FEFLOW-converter.
::::

::::{tab-set}
:::{tab-item} Linux
:sync: unix
Depending on the Linux distribution used, different steps are possible to install FEFLOW.
The DHI supports the installation of FEFLOW on Ubuntu.
Instructions for installing from an apt repository can be found on their [website](https://download.feflow.com/download/FEFLOW/linux/).
Additionally, the environment variables must be set correctly to find the FEFLOW installation in Python.
The following three variables must be set, depending on the version:

```bash
export PYTHONPATH=/opt/feflow/8.0/python
export LD_LIBRARY_PATH=/opt/feflow/8.0/lib64:/opt/feflow/common/qt/lib64:/opt/feflow/common/lib64
export FEFLOW80_ROOT=/opt/feflow/8.0
```

:::
:::{tab-item} macOS
:sync: unix
DHI, the developer of FEFLOW, do not support macOS.
So it is needed, to use the [Docker container](./docker.md) or a virtual machine.
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

In addition, it may be used as Python package.
Further information can be found at: [](../reference/ogstools.feflowlib).

## Example

Following examples demonstrate the usage of the feflowlib::

- [](../auto_examples/howto_conversions/plot_A_feflowlib_start.rst)
- [](../auto_examples/howto_conversions/plot_B_feflowlib_BC_mesh.rst)
- [](../auto_examples/howto_conversions/plot_C_feflowlib_prj.rst)
- [](../auto_examples/howto_conversions/plot_D_feflowlib_CT_simulation.rst)
- [](../auto_examples/howto_conversions/plot_E_feflowlib_H_simulation.rst)
- [](../auto_examples/howto_conversions/plot_F_feflowlib_HT_simulation.rst)
