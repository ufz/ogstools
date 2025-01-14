"""
Setting initial properties and variables in bulk meshes
=======================================================

.. sectionauthor:: Jörg Buchwald (Helmholtz Centre for Environmental Research GmbH - UFZ)

For this example we use an existing mesh file and add the required arrays.
"""

# %%
import numpy as np

import ogstools as ot
from ogstools import examples

# %%
# Setting up material ids and other material properties
# -----------------------------------------------------
#
# A bulk mesh file is loaded. If it already contains old Material IDs
# we delete them first.

mesh = ot.Mesh(examples.mechanics_vtu)
mesh.cell_data.remove("MaterialIDs")


# %% [markdown]
# One way to set-up new Material IDs is by defining a function.
def material_ids(pt: np.array) -> int:
    if np.sqrt((pt[0] - 150) ** 2 + (pt[1] + 650) ** 2 + pt[2] ** 2) < 90:
        return 0
    return 1


# %% [markdown]
# The function is then used to create an array of type np.int32 and same
# size as the number of elements in the mesh.
# The numpy array is later implicitly converted into a VTK array of
# VTK-type Int32 which is the required by OpenGeoSys for Material IDs.
ccp = mesh.cell_centers().points
mat_ids = np.array([material_ids(pt) for pt in ccp], dtype=np.int32)

# %% [markdown]
# Then the array is attached to the mesh.

mesh.cell_data.set_array(mat_ids, "MaterialIDs")

fig = mesh.plot_contourf(ot.variables.material_id)

# %% [markdown]
# Likewise material properties / MeshElement parameters can be set
# in the mesh using standard numpy type floating point arrays.
#
# Creating variable/points fields
# -------------------------------
#
# In this example we use the y-coordinates to create a linear pressure gradient
# along the y-axis.
y = mesh.points[:, 1]
p = 1e6 * y / (np.min(y) - np.max(y)) + 4e6


# %%
# The array is added to the mesh.
mesh.point_data.set_array(p, "pressure_gradient")

fig = mesh.plot_contourf("pressure_gradient")

# %%
# This also works with multidimensional data.
eps = np.zeros((len(mesh.points), 4))
mesh.point_data.set_array(eps, "epsilon_0")

# %% [markdown]
# Mapping element data to node data
# ---------------------------------
# First, we convert the cell data to point data.
# Subsequently, a new array is created and filled point-by-point.
mat_ids = mesh.cell_data_to_point_data().point_data["MaterialIDs"]
p0 = np.where(mat_ids.astype(int) == 0, -1.2e8, -1e6 * mesh.points[:, 1])

# %%
# Finally, the array is added to the mesh and can be
# saved to disc using pyvista/meshio.
mesh.point_data.set_array(p0, "initial_pressure")

# import pyvista as pv
#
# pv.save_meshio("bulk_mesh.vtu", mesh)

fig = mesh.plot_contourf("initial_pressure")
