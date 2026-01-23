"""
Creating meshes from pyvista surfaces
======================================

For this example we create meshes from pyvista surfaces. See other examples for
different meshing algorithms.
"""

from itertools import pairwise
from shutil import which

import ogstools as ot
from ogstools.mesh import create

# an optional requirement (needs to be installed on system or via pip)
tetgen_present = which("tetgen") is not None

if tetgen_present:
    bounds = (-200, 210, -200, 210)
    args = {"bound2D": bounds, "amplitude": 100, "spread": 100, "n": 40}
    gaussians = [
        create.Gaussian2D(**args, height_offset=h) for h in [0, -100, -200]
    ]
    surfaces = [create.Surface(g, mat_id) for mat_id, g in enumerate(gaussians)]
    layers = [create.Layer(sf1, sf2) for sf1, sf2 in pairwise(surfaces)]
    ls = create.LayerSet(layers)

    mesh = ls.to_region_tetrahedron(40).mesh
else:
    mesh = None

# %%
# Visualize the prism mesh
if tetgen_present:
    plotter = ot.plot.contourf(mesh, ot.variables.material_id)
    plotter.show()

# %%
