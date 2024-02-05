"""
Creating meshes from pyvista surfaces
======================================

.. sectionauthor:: Tobias Meisel (Helmholtz Centre for Environmental Research GmbH - UFZ)

For this example we create meshes from pyvista surfaces.
"""

# %%
from ogstools.meshlib.boundary import Layer
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.boundary_subset import Gaussian2D, Surface
from ogstools.meshlib.region import to_region_tetraeder

# See other examples for different meshing algorithms

# %%
# Define a simple surface
bounds = (-200, 210, -200, 210)
args = {"bound2D": bounds, "amplitude": 100, "spread": 100, "n": 40}
surface1 = Surface(Gaussian2D(**args, height_offset=0), material_id=0)
surface2 = Surface(Gaussian2D(**args, height_offset=-100), material_id=1)
surface3 = Surface(Gaussian2D(**args, height_offset=-200), material_id=2)

ls = LayerSet([Layer(surface1, surface2), Layer(surface2, surface3)])
mesh = to_region_tetraeder(ls, 40).mesh
# interprets values as categories
mesh["regions"] = [str(m) for m in mesh["MaterialIDs"]]

# %%
# Visualize the prism mesh
mesh.plot(scalars="regions", show_edges=True)
