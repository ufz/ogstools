"""
Creating meshes from pyvista surfaces
======================================

For this example we create meshes from pyvista surfaces. See other examples for
different meshing algorithms.
"""

# Define a simple surface
from shutil import which

import ogstools as ot
from ogstools import meshlib as ml

tetgen_present = (
    which("tetgen") is not None
)  # an optional requirement (needs to be installed on system or via pip)

if tetgen_present:

    bounds = (-200, 210, -200, 210)
    args = {"bound2D": bounds, "amplitude": 100, "spread": 100, "n": 40}
    surface1 = ml.Surface(ml.Gaussian2D(**args, height_offset=0), material_id=0)
    surface2 = ml.Surface(
        ml.Gaussian2D(**args, height_offset=-100), material_id=1
    )
    surface3 = ml.Surface(
        ml.Gaussian2D(**args, height_offset=-200), material_id=2
    )

    ls = ml.LayerSet(
        [ml.Layer(surface1, surface2), ml.Layer(surface2, surface3)]
    )

    mesh = ml.to_region_tetraeder(ls, 40).mesh
else:
    mesh = None

# %%
# Visualize the prism mesh
if tetgen_present:
    plotter = ot.plot.contourf(mesh, ot.variables.material_id)
    plotter.show()

# %%
