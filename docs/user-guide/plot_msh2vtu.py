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

import ogstools as ot
from ogstools import examples

model_dir = Path(mkdtemp())
msh = examples.msh_geolayers_2d

# %% [markdown]
# Conversion
# ----------
#
# Using :func:`~ogstools.meshlib.gmsh_converter.meshes_from_gmsh` we can
# generate unstructured grids from a given .msh file. As OGS wants to have the
# MaterialIDs numbered beginning from zero, you usually want to set reindex to
# True. The return value is a dict with the mesh names pointing to the
# corresponding meshes. Here we print the filenames and the number of cells. For
# a purely file based approach, OGSTools also provides `msh2vtu` as a command
# line application. Please, call it with `--help` for info about the usage.

# %%
meshes = ot.meshes_from_gmsh(filename=msh, reindex=True, log=False)
print(*[f"{name}: {mesh.n_cells=}" for name, mesh in meshes.items()], sep="\n")

# %% [markdown]
# Let's plot the domain mesh and mark the subdomains.

# %%
domain = meshes["domain"]
fig = ot.plot.contourf(domain, ot.variables.material_id, show_edges=False)

style = {"size": 32, "backgroundcolor": "lightgrey", "ha": "center"}
for name, mesh in meshes.items():
    text_xy = [mesh.center[0], 0.5 * (mesh.center[1] + mesh.bounds[2])]
    fig.axes[0].annotate(name.split("_")[-1], text_xy, **style)

# %% [markdown]
# Note regarding saving
# ---------------------
#
# If you want to save the meshes to be used in a OGS simulation, make sure
# to save the meshes in the following way, so that they are conforming to OGS
# standards:

# %%
for name, mesh in meshes.items():
    pv.save_meshio(Path(model_dir, name + ".vtu"), mesh)
