# fe2vtu - FEFLOW converter

```{eval-rst}
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)
```
## Introduction

The converter is used to convert data stored in FEFLOW binary format to VTK format.
This converter was developed in the Python language and interacts with the Python API of FEFLOW.
`fe2vtu` is a command line interface of the converter that summarizes the main functions to provide the user with an accessible application.
To use the converter, FEFLOW must be installed.

## Installation

Depending on the operating system (OS), different steps are needed for the installation required by the FEFLOW Python API.
However, it is independent of the OS to use a `Docker container` whose setup is predefined in a Dockerfile.
Inside the DOCKER container, the converter works, as in the process of building the container FEFLOW and the required Python packages are installed.
The instructions for creating and running the container are included in the [container repository](https://gitlab.opengeosys.org/owf/first-project-phase/feflow-python-docker).

::::{tab-set}

:::{tab-item} Linux & macOS
:sync: unix

Different steps are possible to install FEFLOW, depending on the used Linux distribution.
We recommend to install packages from the [8.0.1-release](https://download.feflow.com/download/FEFLOW/linux/8.0/u01/deb22/).
Additionally, the environmental variables need to be set correctly, in order to find the FEFLOW installation in `Python`.
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
So, it is needed to use the docker container or a virtual machine.

:::

:::{tab-item} Windows
:sync: win

FEFLOW is fully supported on `Windows`.
The installation is according to the offical [website](https://www.mikepoweredbydhi.com/).

:::

::::

## Requirements:

- FEFLOW
- Python packages:
  - ifm_contrib
  - pyvista

The converter works with [ifm_contrib](https://github.com/red5alex/ifm_contrib), a library for extending the functionalities of the FEFLOW Python API.
This library must be installed beforehand:

```bash
pip install https://github.com/red5alex/ifm_contrib/archive/refs/heads/master.zip
```
## Command line usage

```{argparse}
---
module: ogstools.fe2vtu._cli
func: parser
prog: fe2vtu
---
```

## API usage
Link to api reference

## Example
Link to example