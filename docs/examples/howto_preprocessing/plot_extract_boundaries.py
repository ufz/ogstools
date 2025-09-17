"""
Extracting boundaries of a 2D mesh
==================================

There are situations, when you have a 2D domain mesh, but not the corresponding
boundary meshes (e.g. when extracting a slice from a 3D model). But you need
them to properly set boundary conditions. For those cases `ogstools` provides a
function to generate the individual boundary meshes from the domain mesh or from
a contiuous boundary mesh.
"""

# %%


import ogstools as ot
from ogstools import examples

domain = examples.load_meshseries_THM_2D_PVD()[0]

# %% [markdown]
# We can generate the boundary meshes from the given example in the following
# way and get a dictionary of name and mesh pairs per edge. For details, have a
# look into the documentation:
# :class:`~ogstools.meshlib.meshes.Meshes`. :meth:`~ogstools.meshlib.meshes.Meshes.from_mesh`.

# %%
meshes = ot.Meshes.from_mesh(domain)
boundaries = meshes.subdomains()
for name, mesh in boundaries.items():
    print(name, mesh)


# %% [markdown]
# Let's display and save them:

# %%
fig = domain.plot_contourf(ot.variables.material_id)
colors = ["black", "grey", "lime", "yellow"]
for i, (name, mesh) in enumerate(boundaries.items()):
    ot.plot.line(mesh, ax=fig.axes[0], lw=2, annotate=name, color=colors[i])

# %%
meshes.save()  # optional provide a path
