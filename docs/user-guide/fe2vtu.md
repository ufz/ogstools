# fe2vtu - FEFLOW converter

```{eval-rst}
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

## Introduction

The converter is used to convert data stored in FEFLOW binary format to VTK format (`.vtu`).
This converter was developed in the Python language and interacts with the Python API of FEFLOW.
`fe2vtu` is a command line interface of the converter that summarizes the main functions to provide the user with an accessible application.
To use the converter, FEFLOW-packages (ifm- and Python-package) must be installed.

## Installation

The converter requires FEFLOW to be installed.
Therefore we supply the installation of the necessary FEFLOW packages with `ogstools`.
But the installation of the FEFLOW converter with `ogstools` is optional.
The option can be be selected with the following `pip`-command.

```
pip install ogstools[feflow]
```

## Requirements

- Running via [Docker](./docker.md)

**OR**:

- FEFLOW
- ogstools with feflow (`pip install ogstools[feflow]`)

The converter uses [ifm_contrib](https://github.com/red5alex/ifm_contrib), a library for extending the functionalities of the FEFLOW Python API.

## Command line usage

```{argparse}
---
module: ogstools.fe2vtu._cli
func: parser
prog: fe2vtu
---
```

## API usage

In addition, it may be used as Python module.
Further information can be found at: [](../reference/ogstools.fe2vtu).

## Example

An example of how the API can be used is given at: [](../auto_examples/howto_fe2vtu/index).
