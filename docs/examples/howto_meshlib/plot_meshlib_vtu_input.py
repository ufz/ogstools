"""
Creating meshes from vtu surface files
======================================

.. sectionauthor:: Tobias Meisel (Helmholtz Centre for Environmental Research GmbH - UFZ)

For this example we create meshes from surface layers.
"""

# %%
from pathlib import Path  # To get example vtu files

import numpy as np  # For visualization only

import ogstools.meshplotlib as mpl  # For visualization only
from ogstools.definitions import ROOT_DIR  # To get example vtu files
from ogstools.meshlib.boundary import Layer
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.boundary_subset import Surface
from ogstools.meshlib.region import (
    to_region_prism,
    to_region_simplified,
    to_region_tetraeder,
    to_region_voxel,
)

mpl.setup.reset()
mpl.setup.length.output_unit = "km"
mpl.setup.aspect_limits = [0.2, 5.0]

# %%
# The loaded surfaces are defined within VTU files and adhere to properties such as non-intersecting boundaries with consistent x and y bounds. Alternatively, surfaces can also be created using PyVista with the same properties.


surface_dir = ROOT_DIR / "meshlib/tests/data/mesh1/surface_data"

surface1 = Surface(Path(surface_dir / "00_KB.vtu"), material_id=0)
surface2 = Surface(Path(surface_dir / "01_q.vtu"), material_id=5)
surface3 = Surface(Path(surface_dir / "02_krl.vtu"), material_id=2)
surface4 = Surface(Path(surface_dir / "03_S3.vtu"), material_id=3)

# %%
# Create 3 layers from previously defined surfaces and add all layers to a layerset (ordererd from top to bottom)
layer1 = Layer(top=surface1, bottom=surface2, num_subdivisions=2)
layer2 = Layer(top=surface2, bottom=surface3, num_subdivisions=1)
layer3 = Layer(top=surface3, bottom=surface4, num_subdivisions=0)

layer_set1 = LayerSet(layers=[layer1, layer2, layer3])

# %%
# From layerset creation of simplified meshes (sm), prism meshes (pm), voxel meshes (vm), tetraeder mesh (tm) is possible.
sm = to_region_simplified(layer_set1, xy_resolution=200, rank=3).mesh
pm = to_region_prism(layer_set1, resolution=200).mesh
vm = to_region_voxel(layer_set1, resolution=[200, 200, 50]).mesh
tm = to_region_tetraeder(layer_set1, resolution=200).mesh


# %%
# Visualize the prism mesh

mesh = pm
slices = np.reshape(mesh.slice_along_axis(n=4, axis="y"), (-1, 1))
fig = mpl.plot(slices, "MaterialIDs")
for ax, slice in zip(fig.axes, np.ravel(slices)):
    ax.set_title(f"z = {slice.center[2]:.1f} {mpl.setup.length.data_unit}")

# %%
# Visualize meshes with different meshing algorithm

meshes = [sm, vm, pm, tm]
names = [
    "to_region_simplified",
    "to_region_voxel",
    "to_region_prism",
    "to_region_tetraeder",
]

x_slices = np.reshape([mesh.slice("x") for mesh in meshes], (-1, 1))
fig = mpl.plot(x_slices, "MaterialIDs")
for ax, name in zip(fig.axes, names):
    ax.set_title(name)
