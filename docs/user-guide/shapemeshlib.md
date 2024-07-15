## Introduction

This tool is to create 2D triangular meshes within an area that is defined by a shapefile.

## Requirements

- meshio
- pandamesh
  - (*amnong others*) triangle and gmsh
- geopandas

## Installation

All required python packages can be installed via pip.
But there is a problem with the installation of `pandamesh`.
`pandamesh` requires `triangle`, whose installation via pip fails.
Therefore one needs to build it from source:

```bash
git clone https://github.com/drufat/triangle.git
cd triangle
git submodule update --init
python setup.py install
```

After this `pandamesh` can be installed via pip:

```bash
pip install pandamesh
```

## Usage

`shp2mesh.py` can be used from the commandline:

```bash
python shp2mesh.py [-h] [-i INPUT] [-o OUTPUT] [-c CELLSIZE] [{Triangle,GMSH}] [{simplified,original}]
```

More detailed description of the arguments:

```
positional arguments:
  {Triangle,GMSH}
  Either Triangle or GMSH can be chosen for meshing.
  (default: triangle)

  {simplified,original}
    Either the shapefiles are kept unchanged or they can be simplified.
  (default: original)

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The path to the input shape-file.
  -o OUTPUT, --output OUTPUT
                        The path to the output file.
                        The extension defines the format according to meshio
  -c CELLSIZE, --cellsize CELLSIZE
                        The cellsize for the mesh.
```
