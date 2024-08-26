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
ms  # print MeshSeries meta information


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
# To get a single mesh at a specified timestep. Read data is cached.

mesh_ts10 = ms.mesh(timestep)

# The mesh taken from a specific time step of the mesh series is a pyvista mesh
# Here we use pyvista functionality plot.

mesh_ts10.plot(show_edges=True)


# %%
# You can select a time steps with the `[] operator`
# This example shows the last time step (result) and shows meta information about the mesh.

print(ms[-1])


# %%
# MeshSeries from PVD file
# =========================
ms = examples.load_meshseries_THM_2D_PVD()
ms.mesh(0).plot()


# %%
# Accessing Variables
# ====================
# A MeshSeries provides access to all values of variable at all time steps.
#
# 1. mesh
#    - Get a PyVista mesh at a specific time step and use PyVista functions (e.g., `cell_data <https://docs.pyvista.org/api/core/_autosummary/pyvista.dataset.cell_data>`_).
#    - Efficient for a small set of timesteps, but all data is needed.
#    - :py:mod:`ogstools.meshlib.mesh_series.MeshSeries.mesh`
#    - :py:mod:`ogstools.meshlib.mesh_series.MeshSeries.__getitem__`
#
# 2. data[]
#    - Get a specific variable over a specific time range.
#    - Efficient (only XDMF) for a large set of timesteps , but a small amount of cells / points is needed.
#    - :py:mod:`ogstools.meshlib.mesh_series.MeshSeries.data`


#
# Indexing with data()
# --------------------
# `MeshSeries.data("<variable_name>")`` returns an object, that behaves like a
# `Numpy ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_.
# It allows `multidimensional indexing on ndarrays <https://numpy.org/doc/stable/user/basics.indexing.html>`_
# beyond `Python slicing <https://www.geeksforgeeks.org/python-list-slicing/>`_.

# Typically, the first dimension is the time step, second dimension is the number of points/cells,
# and the last dimension is the number of components of the variable.
#
# Be aware that dimensions of length 1 are omitted, obeying to the rules of
# `Indexing on ndarrays <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
# Data does not work for geometry/points or topology/cells).

ms = examples.load_meshseries_HT_2D_XDMF()
# This mesh series has 97 time steps and 190 points.
# Temperature is a scalar, Darcy velocity is a vector with 2 components. Both are defined at points.

# 1. No range for a dimension (just single time step) -> this dimension gets omitted
ms.data("temperature")[1, :]  # shape is (190,)
# 2. Select range with length for a dimension to keep dimension
ms.data("temperature")[1:2, :]  # shape is (1, 190)
# 3. Select all values for all dimensions
ms.data("temperature")[:]  # shape is (97,190)
# 4. Negative indices are allow - here we select last 2 steps
ms.data("darcy_velocity")[-2:, 1:4, :]  # shape is(2, 3, 2)
# 5. Use select to get a specific range of time steps
temp_on_some_point = ms.data("temperature")[1:3, 2:5]  # shape is (2,3)
print(
    f"Temperature at time steps 1 and 2 for points 2, 3 and 4: {temp_on_some_point}"
)

# %%
# Values function
# ---------------
#
# Convenience function to get all values of a variable.
# See :py:mod:`ogstools.meshlib.mesh_series.MeshSeries.values`.
dv_all = ms.values("darcy_velocity")
print(
    f"Shape of darcy_velocity (time steps, num of points, x and y): {dv_all.shape}"
)
# temperature is a scalar - the last dimension of length 1 is omitted
t_all = ms.values("temperature")
print(f"Shape of temperature (time steps, num of points): {t_all.shape}")
