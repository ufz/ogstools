"""
Read mesh from file (vtu or xdmf) into pyvista mesh
=====================================================

"""

# %%
from ogstools import examples

# %% [markdown]
# To read your own data as a mesh series you can do:
#
# ..  code-block:: python
#
#   from ogstools.meshlib import MeshSeries
#   mesh_series = MeshSeries("filepath/filename_pvd_or_xdmf")
#

# %%
# MeshSeries takes as mandatory argument a str OR pathlib.Path that represents
# the location of the pvd or xdmf file. Here, we load example data:
ms = examples.load_meshseries_HT_2D_XDMF()


# %%
# Accessing  time values
# =======================
# All time value (in seconds) are within a range (e.g. can be converted to list)
# Python slicing is supported.

print(f"First 3 time values are: {ms.timevalues[:3]}.")

# Accessing a specific time step

timestep = 10

print(f"Time value at step {timestep} is {ms.timevalues[timestep]} s.")

# %%
# Read data is cached. The function read is only slow for each new timestep requested.


mesh_ts10 = ms.read(timestep)

# The mesh taken from a specific time step of the mesh series is a pyvista mesh
# Here we use pyvista functionality plot.

mesh_ts10.plot(show_edges=True)


# %%
# MeshSeries from PVD file
# =========================
ms = examples.load_meshseries_THM_2D_PVD()
ms.read(0).plot()

# %%
