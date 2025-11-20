"""
Creating meshes from pyvista surfaces
======================================

For this example we create meshes from pyvista surfaces. See other examples for
different meshing algorithms.
"""

# Define a simple surface
from shutil import which

import ogstools as ot
from ogstools.mesh import create

tetgen_present = (
    which("tetgen") is not None
)  # an optional requirement (needs to be installed on system or via pip)

if tetgen_present:

    bounds = (-200, 210, -200, 210)
    args = {"bound2D": bounds, "amplitude": 100, "spread": 100, "n": 40}
    surface1 = create.Surface(
        create.Gaussian2D(**args, height_offset=0), material_id=0
    )
    surface2 = create.Surface(
        create.Gaussian2D(**args, height_offset=-100), material_id=1
    )
    surface3 = create.Surface(
        create.Gaussian2D(**args, height_offset=-200), material_id=2
    )

    ls = create.LayerSet(
        [create.Layer(surface1, surface2), create.Layer(surface2, surface3)]
    )

    mesh = ls.to_region_tetraeder(40).mesh
else:
    mesh = None

# %%
# Visualize the prism mesh
if tetgen_present:
    plotter = ot.plot.contourf(mesh, ot.variables.material_id)
    plotter.show()

# %%
