# msh2vtu

```{eval-rst}
.. sectionauthor:: Dominik Kern (TU Bergakademie Freiberg)
```

`msh2vtu` is a command line application that converts a Gmsh mesh for use
in OGS by extracting domain-, boundary- and physical group-submeshes and saves
them in vtu-format.

Note that all mesh entities should belong to physical groups.

Supported element types:

- lines (linear and quadratic) in 1D
- triangles and quadrilaterals (linear and quadratic) in 2D
- tetra- and hexahedrons (linear and quadratic) in 3D

## Command line usage

```{argparse}
---
module: ogstools.msh2vtu._cli
func: argparser
prog: msh2vtu
---
```

## API usage

In addition, it may be used as Python module:

```python
from ogstools.msh2vtu import msh2vtu

msh2vtu(
    input_filename="my_mesh.msh",
    output_path="",
    output_prefix="my_meshname",
    dim=0,
    delz=False,
    swapxy=False,
    rdcd=True,
    ogs=True,
    ascii=False,
    log_level="DEBUG",
)
```

______________________________________________________________________

## Examples

A geological model (2D) of a sediment basin by Christian Silbermann and a
terrain model (3D) from the official Gmsh tutorials (x2).

`msh2vtu example/geolayers_2d.msh` generates the following output files:

- *geolayers_2d_boundary.vtu*
- *geolayers_2d_domain.vtu*
- *geolayers_2d_physical_group_RockBed.vtu*
- *geolayers_2d_physical_group_SedimentLayer1.vtu*
- *geolayers_2d_physical_group_SedimentLayer2.vtu*
- *geolayers_2d_physical_group_SedimentLayer3.vtu*
- *geolayers_2d_physical_group_Bottom.vtu*
- *geolayers_2d_physical_group_Left.vtu*
- *geolayers_2d_physical_group_Right.vtu*
- *geolayers_2d_physical_group_Top.vtu*
