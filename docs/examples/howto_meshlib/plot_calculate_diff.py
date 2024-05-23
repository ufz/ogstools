"""
Differences between meshes
==========================

.. sectionauthor:: Feliks Kiszkurno (Helmholtz Centre for Environmental Research GmbH - UFZ)

This example explains how to use functions from meshlib to calculate differences
between meshes.
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_path = '_static/examples/meshlib/plot_calculate_diff_thumbnail.png'

# sphinx_gallery_end_ignore

from ogstools import examples
from ogstools.meshlib import difference, difference_matrix, difference_pairwise
from ogstools.propertylib import properties

mesh_property = properties.temperature

# %%
# 0. Introduction
# ---------------
# This example shows how to calculate differences between meshes.
# It is possible to calculate the difference between multiple meshes at the same
# time. Multiple meshes can be provided either as list or Numpy arrays.
# 4 ways of calculating the difference are presented here.

# %%
# 1. Difference between two meshes
# --------------------------------
# The simplest case is calculating the difference between two meshes. For this
# example, we read two different timesteps from a MeshSeries. It is not required
# that they belong to the same MeshSeries object. As long, as the meshes share
# the same topology and contain the mesh_property of interest, the difference
# will work fine.
mesh_series = examples.load_meshseries_THM_2D_PVD()
mesh1 = mesh_series.read(0)
mesh2 = mesh_series.read(-1)

# %% [markdown]
# The following call will return a mesh containing the difference of the
# mesh_property between the two provided meshes:
#
# .. math::
#
#   \text{Mesh}_1 - \text{Mesh}_2
#

mesh_diff = difference(mesh1, mesh2, mesh_property)

# %% [markdown]
# This returned object will be a PyVista UnstructuredGrid object:

# %%
print(f"Type of mesh_diff: {type(mesh_diff)}")

# %% [markdown]
# 2. Pairwise difference
# ----------------------
# In order to calculate pairwise differences, two lists or arrays of equal
# length have to be provided as the input. They can contain an arbitrary number
# of different meshes, as long as their length is equal.
#
# Consider the two following lists:
#
# .. math::
#
#   \text{List}_{1} = \begin{bmatrix} A_{1} & B_{1} & C_{1}\\ \end{bmatrix}
#
# and
#
# .. math::
#
#   \text{List}_{2} = \begin{bmatrix} A_{2} & B_{2} & C_{2}\\ \end{bmatrix}
#
# The output array will contain following differences between meshes:
#
# .. math::
#
#   \begin{bmatrix} A_{1}-A_{2} & B_{1}-B_{2} & C_{1}-C_{2}\\ \end{bmatrix}
#
# and will have the same shape as the input lists. As in the example below:

meshes_1 = [mesh1] * 3
meshes_2 = [mesh2] * 3

mesh_diff_pair_wise = difference_pairwise(meshes_1, meshes_2, mesh_property)

# %%
print(f"Length of mesh_list1: {len(meshes_1)}")

# %%
print(f"Length of mesh_list2: {len(meshes_2)}")

# %%
print(f"Shape of mesh_diff_pair_wise: {mesh_diff_pair_wise.shape}")

# %% [markdown]
# 3. Matrix difference - one array
# --------------------------------
# If only one list or array is provided, the differences between every possible
# pair of objects in the array will be returned.
#
# Consider following list:
#
# .. math::
#
#   \text{List} = \begin{bmatrix} A & B & C\\ \end{bmatrix}
#
# The output array will contain following differences between meshes:
#
# .. math::
#
#   \begin{bmatrix} A-A & B-A & C-A\\ A-B & B-B & C-B \\ A-C & B-C & C-C \\ \end{bmatrix}
#
# and will have shape of (len(mesh_list), len(mesh_list)). As in the following
# example:

mesh_list = [mesh1, mesh2, mesh1, mesh2]

mesh_diff_matrix = difference_matrix(mesh_list, mesh_property=mesh_property)

# %%
print(f"Length of mesh_list1: {len(mesh_list)}")

# %%
print(f"Shape of mesh_list1: {mesh_diff_matrix.shape}")

# %% [markdown]
# 4. Matrix difference - two arrays of different length
# -----------------------------------------------------
# Unlike difference_pairwise(), difference_matrix() can accept two lists/arrays
# of different lengths. As in Section 3, the differences between all possible
# pairs of meshes in both lists/arrays will be calculated.
#
# Consider following lists:
#
# .. math::
#
#   \text{List}_1 = \begin{bmatrix} A_1 & B_1 & C_1\\ \end{bmatrix}
#
# .. math::
#
#   \text{List}_2 = \begin{bmatrix} A_2 & B_2 \\ \end{bmatrix}
#
# The output array will contain following differences between meshes:
#
# .. math::
#
#   \begin{bmatrix} A_1-A_2 & A_1-B_2 \\ B_1-A_2 & B_1-B_2 \\ C_1-A_2 & C_1-B_2 \\ \end{bmatrix}
#
# and will have a shape of (len(mesh_list_matrix_1), len(mesh_list_matrix_2)).
# As in the following example:

mesh_list_matrix_1 = [mesh1, mesh2, mesh1]
mesh_list_matrix_2 = [mesh2, mesh1]

mesh_diff_matrix = difference_matrix(
    mesh_list_matrix_1, mesh_list_matrix_2, mesh_property
)

# %%
print(f"Length of mesh_list_matrix_1: {len(mesh_list_matrix_1)}")

# %%
print(f"Length of mesh_list_matrix_2: {len(mesh_list_matrix_2)}")

# %%
print(f"Shape of mesh_diff_matrix: {mesh_diff_matrix.shape}")
