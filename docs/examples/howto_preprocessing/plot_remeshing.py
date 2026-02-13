"""
Remeshing with triangle elements
================================

This short example showcases the function ``remesh_with_triangles`` which allows
us to take an existing mesh and re-discretize it with triangle elements. This is
useful for models, where the underlying meshing script is not available or hard
to adapt.
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore

import ogstools as ot
from ogstools import examples

mesh = examples.load_meshseries_THM_2D_PVD().mesh(1)

# %% [markdown]
# This is our example mesh which we want to discretize with triangles.

# %%
fig = ot.plot.contourf(mesh, ot.variables.material_id)

# %% [markdown]
# Here, we do the remeshing and convert the resulting msh file to an
# OGS-compatible vtu file. We can also specify local refinement.

# %%
mesh = examples.load_meshseries_THM_2D_PVD().mesh(1)
repo = (
    mesh.threshold(12, "MaterialIDs")
    .extract_feature_edges()
    .clip_box([2800, 3900, -860, 0, 6.7e3, 6.7e3], invert=False)
)
local_ref = {"pts": repo.points, "SizeMin": 10, "SizeMax": 100, "DistMax": 200}
ref = {"SizeMin": 100}
msh_path = ot.gmsh_tools.remesh_with_triangles(
    mesh, refinement=ref, local_ref=local_ref
)
meshes = ot.Meshes.from_gmsh(msh_path, reindex=False, log=False)
fig = ot.plot.contourf(meshes["domain"], ot.variables.material_id)

# %%
