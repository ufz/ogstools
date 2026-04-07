"""
Modifying integration point data in bulk meshes
===============================================
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = -2

# sphinx_gallery_end_ignore


import numpy as np

import ogstools as ot
from ogstools import examples

# %%
# Looking at the original integration point data
# ==============================================
#
# A bulk mesh file is loaded. It already contains IP data.

mesh = ot.mesh.read(examples.mechanics_2D)
ip_data = ot.mesh.IPdata(mesh)
with np.printoptions(precision=2):
    print(ip_data.info)


# %%
# Let's visualize the trace of the integration point stresses on an artificial
# mesh where each integration point corresponds to one cell.

sigma_ip = ot.variables.stress.replace(
    data_name="sigma_ip", output_name="IP_stress"
)
ip_mesh = ot.mesh.to_ip_mesh(mesh)
fig = ot.plot.contourf(ip_mesh, sigma_ip.trace)

# %%
# There are multiple ways in which you can modify the integration point data.

# %%
# Deleting existing integration point data
# ----------------------------------------

del ip_data["epsilon_m_ip"]

# %%
# Adding new integration point data
# ---------------------------------

const = np.full(ip_mesh.n_cells, 0.0)
isotropic = np.full((ip_mesh.n_cells, 4), [1, 1, 1, 0])
ip_data.set("new_scalar", order=2, num_components=1, values=const)
ip_data.set("new_tensor", order=2, num_components=4, values=isotropic)

# %%
# Modifying existing integration point data
# ------------------------------------------

ip_data["epsilon_ip"].values *= 0

# %%
# A more complex example: We reduce the first three components of the
# integration point stresses close to the hole by using a mask based on the
# distance to the hole's center.

pts = ip_mesh.cell_centers().points
mask = np.hypot(pts[:, 0] - 150, pts[:, 1] + 650) < 90
ip_data["sigma_ip"].values[mask, :3] -= 2.1e6

# %%
# Modify integration point data of a material group
# -------------------------------------------------

# %%
# Sometimes it might be needed to modify only the integration point data of a
# specific material group. Here, top part of the mesh has material id 0, while
# the rest has id 1. We now modify the horizontal integration point stress of
# the top material.

ip_cloud = ot.mesh.to_ip_point_cloud(mesh)
cell_map = mesh.find_containing_cell(ip_cloud.points)
ip_mat_ids = mesh["MaterialIDs"][cell_map]
ip_data["sigma_ip"].values[ip_mat_ids == 0, 0] -= 10e6

# %%
# We can see, that the above modifications are reflected in the mesh.

with np.printoptions(precision=2):
    print(ip_data.info)

# %%
fig = ot.plot.contourf(ot.mesh.to_ip_mesh(mesh), sigma_ip.trace)

# %%
# Remove a material group
# -----------------------

# %%
# The following code recipe removes a material group from the mesh and updates
# the integration point data accordingly.

modified = mesh.threshold([1, 1], scalars="MaterialIDs")
cell_map = modified.find_containing_cell(ip_cloud.points)
# -1 marks those points, where no cell ws found, i.e. those which got removed.
mask = cell_map[cell_map != -1]
for data in ot.mesh.IPdata(modified).values():
    data.values = data.values[mask]

# %%
fig = ot.plot.contourf(ot.mesh.to_ip_mesh(modified), sigma_ip.trace)
# %%
