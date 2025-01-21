"""
Calculate differences between meshes
====================================

.. sectionauthor:: Feliks Kiszkurno (Helmholtz Centre for Environmental Research GmbH - UFZ)

This example shows how to calculate differences between meshes.
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = 2
# pylint: disable=wrong-import-position, wrong-import-order
# ruff: noqa: E402,I001

# sphinx_gallery_end_ignore

import ogstools as ot
from ogstools import examples

# sphinx_gallery_start_ignore

import pyvista as pv
from pathlib import Path
from tempfile import mkdtemp


ot.plot.setup.dpi = 75
ot.plot.setup.show_element_edges = True

tmp_dir = Path(mkdtemp())
mesh_path = tmp_dir / "mesh.msh"
vtu_path = tmp_dir / "mesh_domain.vtu"


def custom_mesh(lengths: int, element_order: int, quads: bool) -> ot.Mesh:
    "Creates a custom mesh and runs a Mechanics simulation on it."
    ot.meshlib.rect(
        lengths=lengths,
        n_edge_cells=21,
        structured_grid=quads,
        order=element_order,
        out_name=mesh_path,
    )
    meshes = ot.meshes_from_gmsh(mesh_path, log=False)
    for name, mesh in meshes.items():
        pv.save_meshio(Path(tmp_dir, name + ".vtu"), mesh)

    model = ot.Project(
        output_file=tmp_dir / "default.prj", input_file=examples.prj_mechanics
    )
    if element_order == 2:
        model.replace_text(4, ".//integration_order")
    model.write_input()
    model.run_model(write_logs=True, args=f"-m {tmp_dir} -o {tmp_dir}")
    return ot.MeshSeries(tmp_dir / "mesh.pvd").mesh(-1)


# sphinx_gallery_end_ignore

# %% [markdown]
# Difference between two meshes
# -----------------------------
# The simplest case is calculating the difference between two meshes. For this
# example, we read two different timesteps from a MeshSeries. It is not required
# that they belong to the same MeshSeries object. As long, as the meshes contain
# the variable of interest, the difference will work fine.

# %%
mesh_series = examples.load_meshseries_THM_2D_PVD().scale(spatial=("m", "km"))
mesh1 = mesh_series.mesh(0)
mesh2 = mesh_series.mesh(-1)

# %% [markdown]
# The following call will return a mesh containing the difference of the
# variable between the two provided meshes. Then we plot the difference:

# %%
mesh_diff = mesh1.difference(mesh2, ot.variables.temperature)
fig = mesh_diff.plot_contourf(ot.variables.temperature)

# %% [markdown]
# Difference between two meshes of different topology
# ---------------------------------------------------
# This examples shall demonstrate how we can compare meshes of different size
# and/or topology. To demonstrate this, we create two artificial meshes.

# %%
quad_mesh = custom_mesh(lengths=0.8, element_order=1, quads=True)
tri_mesh = custom_mesh(lengths=1, element_order=2, quads=False)

# %% [markdown]
# They are visibly quite similar, aside from the difference in cell type and
# size:

# %%
fig = ot.plot.contourf([quad_mesh, tri_mesh], ot.variables.stress["xx"])

# %% [markdown]
# To better quantify it we form the difference and plot the result. The base
# mesh of which we subtract from provides the topology to show the differences.
# Thus we see data here on the quad mesh. Be aware that comparing different
# topologies is inherently affected by interpolation.

# %%
diff_mesh = quad_mesh.difference(tri_mesh, ot.variables.stress)
fig = diff_mesh.plot_contourf(ot.variables.stress.difference["xx"])

# %% [markdown]
# Doing it the other way around shows the difference on the tri-mesh. Here, we
# see, that the subtracted mesh was smaller then the base mesh and thus, the
# which are outside of the domain of the second mesh are masked.

# %%
diff_mesh = tri_mesh.difference(quad_mesh, ot.variables.stress)
fig = diff_mesh.plot_contourf(ot.variables.stress.difference["xx"])

# %% [markdown]
# Differences between multiple meshes
# -----------------------------------
# It is possible to calculate the difference between multiple meshes at the same
# time. Multiple meshes can be provided either as list or numpy arrays.
# 4 ways of calculating the difference are presented here.

# %% [markdown]
# Pairwise difference
# -------------------
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
#   \text{List}_{2} = \begin{bmatrix} A_{2} & B_{2} & C_{2}\\ \end{bmatrix}
#
# The output array will contain following differences between meshes:
#
# .. math::
#
#   \begin{bmatrix} A_{1}-A_{2} & B_{1}-B_{2} & C_{1}-C_{2}\\ \end{bmatrix}
#
# and will have the same shape as the input lists. As in the example below:

# %%
meshes_1 = [mesh1] * 3
meshes_2 = [mesh2] * 3

mesh_diff_pair_wise = ot.meshlib.difference_pairwise(
    meshes_1, meshes_2, ot.variables.temperature
)
print(f"Length of mesh_list1: {len(meshes_1)}")
print(f"Length of mesh_list2: {len(meshes_2)}")
print(f"Shape of mesh_diff_pair_wise: {mesh_diff_pair_wise.shape}")

# %% [markdown]
# Matrix difference - one array
# -----------------------------
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

# %%
mesh_list = [mesh1, mesh2, mesh1, mesh2]

mesh_diff_matrix = ot.meshlib.difference_matrix(
    mesh_list, variable=ot.variables.temperature
)
print(f"Length of mesh_list1: {len(mesh_list)}")
print(f"Shape of mesh_list1: {mesh_diff_matrix.shape}")

# %% [markdown]
# Matrix difference - two arrays of different length
# --------------------------------------------------
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

# %%
mesh_list_matrix_1 = [mesh1, mesh2, mesh1]
mesh_list_matrix_2 = [mesh2, mesh1]

mesh_diff_matrix = ot.meshlib.difference_matrix(
    mesh_list_matrix_1, mesh_list_matrix_2, ot.variables.temperature
)
print(f"Length of mesh_list_matrix_1: {len(mesh_list_matrix_1)}")
print(f"Length of mesh_list_matrix_2: {len(mesh_list_matrix_2)}")
print(f"Shape of mesh_diff_matrix: {mesh_diff_matrix.shape}")
