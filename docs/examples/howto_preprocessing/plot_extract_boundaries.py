"""
Extracting boundaries of a 2D mesh
==================================

There are situations, when you have a 2D domain mesh, but not the corresponding
boundary meshes (e.g. when extracting a slice from a 3D model). But you need
them to properly set boundary conditions. For those cases `ogstools` provides a
function to generate the individual boundary meshes from the domain mesh or from
a continuous boundary mesh.
"""

# %%


import ogstools as ot
from ogstools import examples

domain = examples.load_meshseries_THM_2D_PVD()[0]

# %% [markdown]
# We can generate the boundary meshes from the given example in the following
# way and get a dictionary of name and mesh pairs per edge. For details, have a
# look into the documentation:
# :class:`~ogstools.Meshes`. :meth:`~ogstools.Meshes.from_mesh`.

# %%
meshes = ot.Meshes.from_mesh(domain)
for name, mesh in meshes.subdomains.items():
    print(name, mesh)


# %% [markdown]
# Let's display and save them:

# %%
fig = meshes.plot()

# %%
meshes.save()  # optionally, provide a path

# %% [markdown]
# By having the top mesh as a boundary, you can calculate a depth-based water
# column as an initial conditions or for calculations of integrity criteria

# %%
depth = ot.mesh.depth(meshes.domain, meshes["top"])
meshes.domain.point_data["pressure"] = 1000 * 9.81 * depth
fig = ot.plot.contourf(meshes.domain, "pressure")


# %% [markdown]
# If you need to model an excavation or similar, the `Meshes` class provides
# the useful method :meth:`~ogstools.Meshes.remove_material` which removes a
# specified material from the domain and updates the boundary meshes
# accordingly. The following example is only for demonstration and is not meant
# to make practical sense.

# %%
x = domain.cell_centers().points[:, 0]
mat_ids = domain["MaterialIDs"]
mat_ids[(mat_ids <= 3) & (x < 0)] = 99
meshes.remove_material(99)
fig = meshes.plot()

# %%
