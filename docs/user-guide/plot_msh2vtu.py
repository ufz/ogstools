"""
Meshes from gmsh (msh2vtu)
==========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

Here you find, how you can generate OGS-compatible meshes from a .msh file.
This is accomplished by extracting domain-, boundary- and physical groups
from the gmsh mesh. This requires the mesh entities to be assigned to a physical
group in the gmsh mesh.
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv

import ogstools as ogs
from ogstools import examples

model_dir = Path(mkdtemp())

# %% [markdown]
# The usual (file-based) way is to just call msh2vtu, which does the conversion
# and writes the meshes to the disc. It also returns the full filepaths to the
# written meshes. Here we just print the filenames. OGSTools also provides
# `msh2vtu` as a command line application.

# %%
mesh_paths = ogs.msh2vtu(
    examples.msh_geolayers_2d, output_path=model_dir, log=False
)
print(*[mesh_path.name for mesh_path in mesh_paths], sep="\n")

# %% [markdown]
# We can also use a pure python object approach and generate unstructured grids
# from the .msh file. As OGS wants to have the MaterialIDs numbered beginning
# from zero, you usually want to set reindex to True. The following function
# return a dict with the meshes and the corresponding names.

# %%
meshes = ogs.meshlib.meshes_from_gmsh(
    examples.msh_geolayers_2d, reindex=True, log=False
)
print(*meshes.keys(), sep="\n")

# %% [markdown]
# Let's plot the domain mesh and mark the subdomains.

# %%
domain = meshes["geolayers_2d_domain"]

fig = ogs.plot.contourf(domain, ogs.variables.material_id, show_edges=False)

style = {"size": 32, "backgroundcolor": "lightgrey", "ha": "center"}
for name, mesh in meshes.items():
    if "domain" in name:
        continue  # skip annotating for the domain mesh
    text_xy = mesh.center[:2]
    if mesh.area:  # minor adjustment for better visibility
        text_xy[1] += 0.25 * mesh.bounds[2]  # move towards min y value
    fig.axes[0].annotate(name.split("_")[-1], text_xy, **style)

# %% [markdown]
# If you use this approach make sure to save the meshes in the following way,
# so that they are conforming to OGS standards (this is what msh2vtu does):

# %%
for name, mesh in meshes.items():
    pv.save_meshio(Path(model_dir, name + ".vtu"), mesh)

# %% [markdown]
# Whichever version you choose to use, the resulting meshes should be ready to
# use for a simulation. For more information have a look at the function API's:
# :func:`~ogstools.meshlib.gmsh_converter.meshes_from_gmsh` and
# :func:`~ogstools.msh2vtu.converter.msh2vtu`
