# ogs6py

## Overview

ogs6py is a Python-API for the OpenGeoSys finite element software.
Its main functionalities include creating and altering OGS6 input files, as well as executing OGS.
The package allows you to automate OGS-workflows in Python via Jupyter or just plain Python scripts.
This enables the user to create parameter plans and to execute them in parallel.
API: [](../reference/ogstools.ogs6py.rst).

# Features

- alternate existing files (e.g., for parameter sweeps)
- create new input files from scratch
- execute project files
- tailored alteration of input files e.g. for mesh replacements or restarts
- display and export parameter settings

## Example

Following examples demonstrate the usage of the `ogs6py`::

- [](../auto_examples/howto_prjfile/plot_creation.rst)
- [](../auto_examples/howto_prjfile/plot_manipulation.rst)
