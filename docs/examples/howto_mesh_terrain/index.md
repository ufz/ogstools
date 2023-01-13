# Terrain meshing

```{eval-rst}
.. sectionauthor:: Dominik Kern (TU Bergakademie Freiberg)
```

Here we show different tools to mesh a minimum example of a terrain, given by raster data.
These are

- [gmsh](http://gmsh.info)
- [OGS Utilities](https://www.opengeosys.org/docs/tools/getting-started/overview/)
- [PyvVsta](https://docs.pyvista.org/index.html)
- [TetGen](https://wias-berlin.de/software/index.jsp?id=TetGen&lang=1)

![terrain](./terrain.png)

## Gmsh

If the relief is given as relief.grd, then we need to convert it first (file names are hard coded)
`python -m ogstools.msh2vtu.examples.howto_mesh_terrain.grd2stl`.

Running `python -m ogstools.msh2vtu.examples.howto_mesh_terrain.pyvista_mesh` reads in `relief.stl` and meshes the volume between the relief and a *z*-coordinate specified in the script.
In addition it creates physical groups for all bounding surfaces.

![gmsh](gmsh.png)

```{literalinclude} ../../../ogstools/msh2vtu/examples/howto_mesh_terrain/gmsh_mesh.py
```

## OGS Utilities

OGS comes with some utilities for meshing. Required are raster files (`.grd`) and a file specifying their sequence (here `layer_file_list`).
The following commands create a 2D mesh, extrude it to 3D-wedge elements and then fit in a structured hex mesh.

```bash
generateStructuredMesh -e tri --lx 10 --nx 20 --ly 10 --ny 20 -o tri2d_mesh.vtu
createLayeredMeshFromRasters -i tri2d_mesh.vtu -o wedge3d_mesh.vtu -r layer_file_list
Vtu2Grid -i wedge3d_mesh.vtu -o hex3d_mesh.vtu -x 0.4
```

![ogstools](ogstools.png)

## PyVista

PyVista is mainly made for visualization, but brings some meshing functionality.
It has a grid reader to read relief.grd and a meshio-interface to write to relief.vtu.
Running `python -m ogstools.msh2vtu.examples.howto_mesh_terrain.pyvista_mesh` generates a structured mesh on the relief and extrudes it downwards to 3D.

![pyvista](pyvista.png)

```{literalinclude} ../../../ogstools/msh2vtu/examples/howto_mesh_terrain/pyvista_mesh.py
```

## TetGen

TetGen creates a 3D mesh from a 2D mesh on a closed surface.
This surface may be either read from stl-files or PyVista-data.
Currently the *pyvista-tetgen basic example* (`tetgen_example.py`) has been prepared for terrain meshing (`tetgen_mesh.py`), but is not finished yet.

```{literalinclude} ../../../ogstools/msh2vtu/examples/howto_mesh_terrain/tetgen_mesh.py
```
