"""
Extracting boundaries of a 2D mesh
==================================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

There are situations, when you have a 2D domain mesh, but not the corresponding
boundary meshes (e.g. when extracting a slice from a 3D model). But you need
them to properly set boundary conditions. For those cases `ogstools` provides a
function to generate the individual boundary meshes from the domain mesh or from
a contiuous boundary mesh.
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv

import ogstools as ot
from ogstools import examples

domain = examples.load_meshseries_THM_2D_PVD()[0]

# %% [markdown]
# We can generate the boundary meshes from the given example in the following
# way and get a dictionary of name and mesh pairs per edge. For details, have a
# look into the documentation: :py:func:`ogstools.meshlib.extract_boundaries`.

# %%
boundaries = ot.meshlib.extract_boundaries(domain)
for name, mesh in boundaries.items():
    print(name, mesh)


# %% [markdown]
# Let's display and save them:

# %%
fig = domain.plot_contourf(ot.variables.material_id)
colors = ["black", "grey", "lime", "yellow"]
for i, (name, mesh) in enumerate(boundaries.items()):
    ot.plot.line(mesh, ax=fig.axes[0], lw=2, annotate=name, color=colors[i])
    pv.save_meshio(Path(mkdtemp()) / (name + ".vtu"), mesh)

# %%
