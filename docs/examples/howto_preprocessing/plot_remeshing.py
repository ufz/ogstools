"""
Remeshing with triangle elements
================================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

This short example showcases the function ``remesh_with_tri`` which allows us to
take an existing mesh and re-discretize it with triangle elements. This is
useful for models, where the underlying meshing script is not available or hard
to adapt.
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = -1

# sphinx_gallery_end_ignore

from pathlib import Path
from tempfile import mkdtemp

import ogstools as ot
from ogstools import examples

mesh = examples.load_meshseries_THM_2D_PVD().mesh(1)

# %% This is our example mesh which we want to discretize with triangle
# elements.
fig = mesh.plot_contourf(ot.variables.material_id)

# %% [markdown]
# Here, we do the remeshing and convert the resulting msh file to an
# OGS-compatible vtu file.

# %%
mesh = examples.load_meshseries_THM_2D_PVD().mesh(1)
temp_dir = Path(mkdtemp())
msh_path = temp_dir / "tri_mesh.msh"
ot.meshlib.gmsh_meshing.remesh_with_triangles(mesh, msh_path)
meshes = ot.meshes_from_gmsh(msh_path, reindex=False, log=False)
fig = meshes["domain"].plot_contourf(ot.variables.material_id)
