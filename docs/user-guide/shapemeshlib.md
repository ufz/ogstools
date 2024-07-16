# shapemeshlib - shp2mesh

```{eval-rst}
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)
```

## Introduction

`shapemeshlib` is a module to create 2D triangular meshes within an area that is defined by a shapefile.
`shp2mesh` is the corresponding commandline tool that summarizes its feature to create a mesh from a shapefile.

## Features

- Create a mesh from a shapefile
- Choose between `triangle` and `gmsh` for meshing
- Simplify the shapefile before meshing

## Command line usage

```{argparse}
---
module: ogstools.shapemeshlib._cli
func: parser
prog: shp2mesh
---
```
