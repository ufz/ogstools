"""
Creating meshes from vtu surface files
======================================

.. sectionauthor:: Tobias Meisel (Helmholtz Centre for Environmental Research GmbH - UFZ)

For this example we create meshes from surface layers.
"""

# %%
from pathlib import Path  # To get example vtu files

from ogstools.definitions import EXAMPLES_DIR  # To get example vtu files
from ogstools.meshlib.boundary import Layer
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.boundary_subset import Surface
from ogstools.meshlib.region import (
    to_region_prism,
    to_region_simplified,
    to_region_tetraeder,
    to_region_voxel,
)

# %%
# The loaded surfaces are defined within VTU files and adhere to properties such as non-intersecting boundaries with consistent x and y bounds. Alternatively, surfaces can also be created using PyVista with the same properties.


surface_dir = EXAMPLES_DIR / "meshlib" / "mesh1" / "surface_data"

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

# %% [markdown]
# Simplified mesh
# ---------------
# %%
sm["regions"] = [str(m) for m in sm["MaterialIDs"]]
sm.scale([1, 1, 5]).plot(scalars="regions", show_edges=True)

# %% [markdown]
# Voxel mesh
# ---------------
# %%
vm["regions"] = [str(m) for m in vm["MaterialIDs"]]
vm.scale([1, 1, 5]).plot(scalars="regions", show_edges=True)

# %% [markdown]
# Prism mesh
# ---------------
# %%
pm["regions"] = [str(m) for m in pm["MaterialIDs"]]
pm.scale([1, 1, 5]).plot(scalars="regions", show_edges=True)

# %% [markdown]
# Tetraeder mesh
# ---------------
# %%
tm["regions"] = [str(m) for m in tm["MaterialIDs"]]
tm.scale([1, 1, 5]).plot(scalars="regions", show_edges=True)
