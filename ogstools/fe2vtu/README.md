# FEFLOW converter
## Introduction
The converter is used to convert data stored in FEFLOW binary format to VTK format.
For this it is necessary to have FEFLOW installed.
This converter was developed in the Python language and interacts with the Python API of FEFLOW.


## Installation
There are two ways to work with the FEFLOW Python API on Linux:

1. With a Docker container, whose setup is predefined in a Dockerfile.
Inside the DOCKER container, the converter works, as in the process of building the container FEFLOW and the required Python packages are installed.
The instructions for creating and running the container are included in the container ([setup](https://gitlab.opengeosys.org/owf/first-project-phase/feflow-python-docker)) repository.

2. Install FEFLOW and all the required Python packages. Then run the script `fepython` that correctly sets the required environment variables
   * run: `chmod +x fepython80`
   * set the environmental variables: `source fepython80`

In `Windows` the environmental variables should be set correctly with the installation. No further steps are required for the converter to be working.

The converter works with [ifm_contrib](https://github.com/red5alex/ifm_contrib), a library for extending the functionalities of the FEFLOW Python API.
This library must be installed beforehand:
```bash
 pip  install https://github.com/red5alex/ifm_contrib/archive/refs/heads/master.zip
 ```

## Requirements:
* FEFLOW
* Python packages:
    * ifm_contrib
    * pyvista

## Usage
The converter can be used in four different cases.
In each case it needs a Feflow input file and a name for the output file, which should have the extension `*.vtu`. 
The four cases refer to:
1. `geo_surface`: writes the surface of the mesh without -
2. `geometry`: writes the mesh without -
3. `properties`: writes the mesh with -
4. `properties_surface`: writes the surface of the mesh with -

properties on cells and points. 
```bash
fe2vtu [-h] [-i INPUT] [-o OUTPUT] [{geo_surface,geometry,properties,properties_surface}]
```
