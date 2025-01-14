"""
Analyzing integration point data
================================

This examples shall demonstrate how we can better visualize integration point
data (raw data used in OGS's equation system assembly without output related
artefacts), by tesselating elements in such a way that each integration point is
represented by one subsection of a cell.
"""

# %% [markdown]
# For brevity of this example we wrap the entire workflow from meshing and
# simulation to plotting in a parameterized function. The main function of
# importance is :meth:`~ogstools.meshlib.mesh.Mesh.to_ip_mesh`. We will use this
# function to show the tessellated visualization of the integration point data
# for different element and integration point orders and element types. Note,
# you can also tessellate an entire MeshSeries via
# :meth:`~ogstools.meshlib.mesh_series.MeshSeries.ip_tesselated`

from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv

import ogstools as ot
from ogstools import examples
from ogstools.meshlib.gmsh_meshing import rect

ot.plot.setup.dpi = 75
ot.plot.setup.show_element_edges = True

sigma_ip = ot.variables.stress.replace(
    data_name="sigma_ip", output_name="IP_stress"
)

tmp_dir = Path(mkdtemp())
msh_path = tmp_dir / "mesh.msh"


def simulate_and_plot(elem_order: int, quads: bool, intpt_order: int):
    rect(
        lengths=1,
        n_edge_cells=6,
        structured_grid=quads,
        order=elem_order,
        out_name=msh_path,
    )
    meshes = ot.meshes_from_gmsh(msh_path, log=False)
    for name, mesh in meshes.items():
        pv.save_meshio(Path(tmp_dir, name + ".vtu"), mesh)

    model = ot.Project(
        output_file=tmp_dir / "default.prj",
        input_file=examples.prj_mechanics,
    )
    model.replace_text(intpt_order, xpath=".//integration_order")
    model.write_input()
    model.run_model(write_logs=True, args=f"-m {tmp_dir} -o {tmp_dir}")
    mesh = ot.MeshSeries(tmp_dir / "mesh.pvd").mesh(-1)
    int_pts = mesh.to_ip_point_cloud()
    ip_mesh = mesh.to_ip_mesh()

    fig = mesh.plot_contourf(ot.variables.stress)
    fig.axes[0].scatter(
        int_pts.points[:, 0], int_pts.points[:, 1], color="k", s=10
    )
    fig = ip_mesh.plot_contourf(sigma_ip)
    fig.axes[0].scatter(
        int_pts.points[:, 0], int_pts.points[:, 1], color="k", s=10
    )


# %% [markdown]
# Triangles with increasing integration point order
# -------------------------------------------------
# .. dropdown:: Why does the stress not change with the integration point order?
#
#     In linear triangular finite elements, the shape functions used
#     to interpolate displacements are linear functions of the coordinates.
#     As this is a linear elastic example, the displacements are linear.
#     The strain, which is obtained by differentiating the displacement, will
#     thus be constant throughout the element. The stress, which is related to
#     the strain through a constitutive relationship will also be constant
#     throughout the element. Thus, the stress is not affected by the
#     integration point order.

simulate_and_plot(elem_order=1, quads=False, intpt_order=2)

# %%
simulate_and_plot(elem_order=1, quads=False, intpt_order=3)

# %%
simulate_and_plot(elem_order=1, quads=False, intpt_order=4)

# %% [markdown]
# Quadratic triangles
# -------------------

simulate_and_plot(elem_order=2, quads=False, intpt_order=4)

# %% [markdown]
# Quadrilaterals with increasing integration point order
# ------------------------------------------------------
# .. dropdown:: Why does the stress change here?
#
#     In contrast to triangular elements, quadrilateral elements use bilinear
#     shape functions. Thus, the differentiation of the displacement leads to
#     bilinear strain. The stress in turn is bilinear as well and can change
#     within the elements. The number of integration points consequently
#     affects the resulting stress field.

simulate_and_plot(elem_order=1, quads=True, intpt_order=2)

# %%
simulate_and_plot(elem_order=1, quads=True, intpt_order=3)

# %%
simulate_and_plot(elem_order=1, quads=True, intpt_order=4)

# %% [markdown]
# Quadratic quadrilateral
# -----------------------

simulate_and_plot(elem_order=2, quads=True, intpt_order=4)
