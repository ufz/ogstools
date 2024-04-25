# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""
Creating a BHE mesh (Borehole Heat Exchanger)
=============================================

This example demonstrates how to use gen_bhe_mesh() to create a Borehole Heat Exchanger (BHE) mesh.
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv
from pyvista.plotting import Plotter

from ogstools.meshlib.gmsh_meshing import gen_bhe_mesh
from ogstools.msh2vtu import msh2vtu

# %% [markdown]
# 0. Introduction
# ---------------
# This example shows the general usage of gen_bhe_mesh and how some of the parameters will efect the mesh. This section demonstrates the mesh genration with only three soil layers, groundwater flow in one layer and three BHE's. However, this tool provides multiple soil layers, groundwater flow in multiple layers and multiple BHE's. The mesh sizes provides good initial values for the most Heat-Transport-BHE simulations in OGS. They can also be set by the user, to customize the mesh. Feel free to try it out!

# %% [markdown]
# 1. Create a simple prism mesh
# --------------------------------
# Generate a customizable prism BHE mesh (using gmsh):

# %%
tmp_dir = Path(mkdtemp())
msh_file = tmp_dir / "bhe_prism.msh"
gen_bhe_mesh(
    length=150,
    width=100,
    layer=[50, 50, 50],
    groundwater=[[-30, 1, "+x"]],
    BHE_array=[
        [50, 40, -1, -60, 0.076],
        [50, 50, -1, -60, 0.076],
        [50, 60, -1, -60, 0.076],
    ],
    meshing_type="prism",
    out_name=msh_file,
)

# %% [markdown]
# Now we convert the gmsh mesh to the VTU format with msh2vtu.
# Passing the list of dimensions [1, 3] to msh2vtu ensures, that the BHE line
# elements will also be part of the domain mesh.

# %%
msh2vtu(
    msh_file, output_path=tmp_dir, dim=[1, 3], reindex=True, log_level="ERROR"
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract BHE line:

# %%
mesh = pv.read(tmp_dir / "bhe_prism_domain.vtu")
top_mesh = pv.read(tmp_dir / "bhe_prism_physical_group_Top_Surface.vtu")
bottom_mesh = pv.read(tmp_dir / "bhe_prism_physical_group_Bottom_Surface.vtu")
gw_mesh = pv.read(tmp_dir / "bhe_prism_physical_group_Groundwater_Inflow_0.vtu")
bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)

# %% [markdown]
# Translate the submeshes for visualization only:

# %%
top_mesh = top_mesh.translate((0, 0, 10), inplace=False)
bottom_mesh = bottom_mesh.translate((0, 0, -10), inplace=False)
gw_mesh = gw_mesh.translate((-10, 0, 0), inplace=False)

# %% [markdown]
# Visualize the mesh:

# %%
p = Plotter()
p.add_mesh(mesh, style="wireframe", color="grey")
p.add_mesh(
    mesh.clip("x", bhe_line.center, crinkle=True),
    show_edges=True,
    scalars="MaterialIDs",
    cmap="Accent",
    categories=True,
    scalar_bar_args={"vertical": True, "n_labels": 2, "fmt": "%.0f"},
)
p.add_mesh(bhe_line, color="r", line_width=3)
p.add_mesh(top_mesh, show_edges=True)
p.add_mesh(bottom_mesh, show_edges=True)
p.add_mesh(gw_mesh, show_edges=True)
p.show()

# %% [markdown]
# 1. Create a simple structured mesh
# --------------------------------
# Generate a customizable structured BHE mesh (using gmsh):

# %%
tmp_dir = Path(mkdtemp())
msh_file = tmp_dir / "bhe_structured.msh"
gen_bhe_mesh(
    length=150,
    width=100,
    layer=[50, 50, 50],
    groundwater=[[-30, 1, "+x"]],
    BHE_array=[
        [50, 40, -1, -60, 0.076],
        [50, 50, -1, -60, 0.076],
        [50, 60, -1, -60, 0.076],
    ],
    meshing_type="structured",
    out_name=msh_file,
)

# %% [markdown]
# Now we convert the gmsh mesh to the VTU format with msh2vtu.
# Passing the list of dimensions [1, 3] to msh2vtu ensures, that the BHE line
# elements will also be part of the domain mesh.

# %%
msh2vtu(
    msh_file, output_path=tmp_dir, dim=[1, 3], reindex=True, log_level="ERROR"
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract BHE line:

# %%
mesh = pv.read(tmp_dir / "bhe_structured_domain.vtu")
top_mesh = pv.read(tmp_dir / "bhe_structured_physical_group_Top_Surface.vtu")
bottom_mesh = pv.read(
    tmp_dir / "bhe_structured_physical_group_Bottom_Surface.vtu"
)
gw_mesh = pv.read(
    tmp_dir / "bhe_structured_physical_group_Groundwater_Inflow_0.vtu"
)
bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)

# %% [markdown]
# Translate the submeshes for visualization only:

# %%
top_mesh = top_mesh.translate((0, 0, 10), inplace=False)
bottom_mesh = bottom_mesh.translate((0, 0, -10), inplace=False)
gw_mesh = gw_mesh.translate((-10, 0, 0), inplace=False)

# %% [markdown]
# Visualize the mesh:

# %%
p = Plotter()
p.add_mesh(mesh, style="wireframe", color="grey")
p.add_mesh(
    mesh.clip("x", bhe_line.center, crinkle=True),
    show_edges=True,
    scalars="MaterialIDs",
    cmap="Accent",
    categories=True,
    scalar_bar_args={"vertical": True, "n_labels": 2, "fmt": "%.0f"},
)
p.add_mesh(bhe_line, color="r", line_width=3)
p.add_mesh(top_mesh, show_edges=True)
p.add_mesh(bottom_mesh, show_edges=True)
p.add_mesh(gw_mesh, show_edges=True)
p.show()

# %% [markdown]
# 2. Create an advanced structured mesh
# --------------------------------
# Generate a customizable structured BHE mesh with advanced mesh sizing options (using gmsh). To understand the specific behaviour of every mesh parameter, test each one after another.

# %%
msh_file = tmp_dir / "bhe_structured_advanced.msh"
gen_bhe_mesh(
    length=150,
    width=100,
    layer=[50, 50, 50],
    groundwater=[[-30, 1, "+x"]],
    BHE_array=[
        [50, 40, -1, -60, 0.076],
        [50, 50, -1, -60, 0.076],
        [50, 60, -1, -60, 0.076],
    ],
    meshing_type="structured",
    target_z_size_coarse=10,  # default value 7.5
    target_z_size_fine=2,  # default value 1.5
    n_refinement_layers=1,  # default value 2
    dist_box_x=15,  # default value 10
    propagation=1.2,  # default value 1.1
    inner_mesh_size=8,  # default value 5
    outer_mesh_size=12,  # default value 10
    out_name=msh_file,
)

# %% [markdown]
# Now we convert the gmsh mesh to the VTU format with msh2vtu.
# Passing the list of dimensions [1, 3] to msh2vtu ensures, that the BHE line
# elements will also be part of the domain mesh.

# %%
msh2vtu(
    msh_file, output_path=tmp_dir, dim=[1, 3], reindex=True, log_level="ERROR"
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract BHE line:

# %%
mesh = pv.read(tmp_dir / "bhe_structured_advanced_domain.vtu")
top_mesh = pv.read(
    tmp_dir / "bhe_structured_advanced_physical_group_Top_Surface.vtu"
)
bottom_mesh = pv.read(
    tmp_dir / "bhe_structured_advanced_physical_group_Bottom_Surface.vtu"
)
gw_mesh = pv.read(
    tmp_dir / "bhe_structured_advanced_physical_group_Groundwater_Inflow_0.vtu"
)
bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)

# %% [markdown]
# Translate the submeshes for visualization only:

# %%
top_mesh = top_mesh.translate((0, 0, 10), inplace=False)
bottom_mesh = bottom_mesh.translate((0, 0, -10), inplace=False)
gw_mesh = gw_mesh.translate((-10, 0, 0), inplace=False)

# %% [markdown]
# Visualize the mesh:

# %%
p = Plotter()
p.add_mesh(mesh, style="wireframe", color="grey")
p.add_mesh(
    mesh.clip("x", bhe_line.center, crinkle=True),
    show_edges=True,
    scalars="MaterialIDs",
    cmap="Accent",
    categories=True,
    scalar_bar_args={"vertical": True, "n_labels": 2, "fmt": "%.0f"},
)
p.add_mesh(bhe_line, color="r", line_width=3)
p.add_mesh(top_mesh, show_edges=True)
p.add_mesh(bottom_mesh, show_edges=True)
p.add_mesh(gw_mesh, show_edges=True)
p.show()
