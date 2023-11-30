"""
Read mesh from file (vtu or xdmf) into pyvista mesh
=====================================================

"""

# %%
from ogstools.meshlib import MeshSeries
from ogstools.meshlib.examples import xdmf_file
from ogstools.meshplotlib.examples import THM_2D_file as pvd_file

# %%
# MeshSeries takes as mandatory argument a str OR pathlib.Path that represents the location of the pvd or xdmf file.
print(xdmf_file)
ms = MeshSeries(xdmf_file)


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
ms = MeshSeries(pvd_file)
ms.read(0).plot()

# %%
