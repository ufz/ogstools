"""
Creating meshes from pyvista surfaces
======================================

.. sectionauthor:: Tobias Meisel (Helmholtz Centre for Environmental Research GmbH - UFZ)

For this example we create meshes from pyvista surfaces.
"""

# %%
import numpy as np  # For visualization only

import ogstools.meshplotlib as mpl  # For visualization only
from ogstools.meshlib.boundary import Layer
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.boundary_subset import Gaussian2D, Surface
from ogstools.meshlib.region import (
    to_region_tetraeder,
)

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

# %%
# Visualize the prism mesh

slices = np.reshape(mesh.slice_along_axis(n=4, axis="y"), (1, -1))
mpl.setup.aspect_limits = [0.2, 5.0]
fig = mpl.plot(slices, "MaterialIDs")
for ax, slice in zip(fig.axes, np.ravel(slices)):
    ax.set_title(f"z = {slice.center[2]:.1f} {mpl.setup.length.data_unit}")
_ = fig
