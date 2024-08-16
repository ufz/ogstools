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

print(f"First 3 time values are: {ms.timevalues()[:3]}.")

# Accessing a specific time step

timestep = 10

print(f"Time value at step {timestep} is {ms.timevalues()[timestep]} s.")

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
# Accessing Attributes
# ====================
# A MeshSeries provides access to all values of attributes at all time steps.
#
# 1. read
#    - Get a PyVista mesh at a specific time step and use PyVista functions (e.g., `cell_data <https://docs.pyvista.org/api/core/_autosummary/pyvista.dataset.cell_data>`_).
#    - Efficient for a small set of timesteps, but all data is needed.
#    - :py:mod:`ogstools.meshlib.mesh_series.MeshSeries.read`
#
# 2. select
#    - Get a specific attribute over a specific time range.
#    - Efficient for a large set of timesteps (currently only XDMF), but limited data is needed.
#    - :py:mod:`ogstools.meshlib.mesh_series.MeshSeries.select`


#
# Indexing with the select function
# ---------------------------------
# 1. The select function return an object that allows
# `Python slicing <https://www.geeksforgeeks.org/python-list-slicing/>`_.
# The objects index works like `Indexing on ndarrays <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
#
# The result of a call using `[]` syntax is always a
# `Numpy ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ ,
# Typically, the first dimension is the time step, second dimension is the number of points/cells,
# and the last dimension is the number of components of the attribute.
#
# Be aware that dimensions of length 1 are omitted, obeying to the rules of `Indexing on ndarrays <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
# Select is only working on dynamic attributes (not geometry/points or topology/cells).

ms = examples.load_meshseries_HT_2D_XDMF()

# 1. No range for a dimension (just single time step) -> this dimension gets omitted
ms.select("temperature")[1, :]  # shape is (190,)
# 2. Select range with length for a dimension to keep dimension
ms.select("temperature")[1:2, :]  # shape is (1, 190)
# 3. Select all values for all dimensions
ms.select("temperature")[:]  # shape is (97,190)
# 4. Negative indices are allow - here we select last 2 steps
ms.select("darcy_velocity")[-2:, 1:4, :]  # shape is(2, 3, 2)
# 5. Use select to get a specific range of time steps
temp_on_some_point = ms.select("temperature")[1:3, 2:5]  # shape is (2,3)
print(
    f"Temperature at time steps 1 and 2 for points 2, 3 and 4: {temp_on_some_point}"
)

# %%
# Values function
# ---------------
#
# Convenience function to get all values of an attribute.
# See :py:mod:`ogstools.meshlib.mesh_series.MeshSeries.values`.
dv_all = ms.values("darcy_velocity")
print(
    f"Shape of darcy_velocity (time steps, num of points, x and y): {dv_all.shape}"
)
# temperature is a scalar - the last dimension of length 1 is omitted
t_all = ms.values("temperature")
print(f"Shape of temperature (time steps, num of points): {t_all.shape}")
