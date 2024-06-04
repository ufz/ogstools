# %%
"""
Creating a BHE mesh (Borehole Heat Exchanger)
===================================================
"""

# %% [markdown]
# This example demonstrates how to use :py:mod:`ogstools.meshlib.gmsh_meshing.gen_bhe_mesh` to create
# a Borehole Heat Exchanger (BHE) mesh.

# %%
from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv
from pyvista.plotting import Plotter

from ogstools.meshlib.gmsh_meshing import BHE, Groundwater, gen_bhe_mesh

# %% [markdown]
# 0. Introduction
# ----------------
# This example shows the general usage of :py:mod:`ogstools.meshlib.gmsh_meshing.gen_bhe_mesh` and how some of
# the parameters will effect the mesh. This section demonstrates the mesh
# generation with only three soil layers, groundwater flow in one layer
# and three BHE's. However, this tool provides multiple soil layers,
# groundwater flow in multiple layers and multiple BHE's. The mesh sizes
# provides good initial values for the most Heat-Transport-BHE simulations
# in OGS. They can also be set by the user, to customize the mesh.
# Feel free to try it out!

# %% [markdown]
# 1. Create a simple prism mesh
# --------------------------------
# Generate a customizable prism BHE mesh:

# %%
tmp_dir = Path(mkdtemp())
vtu_file = tmp_dir / "bhe_prism.vtu"
bhe_meshes = gen_bhe_mesh(
    length=150,
    width=100,
    layer=[50, 50, 50],
    groundwater=Groundwater(
        begin=-30, isolation_layer_id=1, flow_direction="+x"
    ),
    BHE_Array=[
        BHE(x=50, y=40, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=50, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=60, z_begin=-1, z_end=-60, borehole_radius=0.076),
    ],
    meshing_type="prism",
    out_name=vtu_file,
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract the BHE lines:

# %%
mesh = pv.read(tmp_dir / bhe_meshes[0])
top_mesh = pv.read(tmp_dir / bhe_meshes[1])
bottom_mesh = pv.read(tmp_dir / bhe_meshes[2])
gw_mesh = pv.read(tmp_dir / bhe_meshes[3])
bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)

# %% [markdown]
# Visualize the mesh:

# %%
top_mesh = top_mesh.translate((0, 0, 10), inplace=False)
bottom_mesh = bottom_mesh.translate((0, 0, -10), inplace=False)
gw_mesh = gw_mesh.translate((-10, 0, 0), inplace=False)

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
p.add_axes()
p.show()

# %% [markdown]
# 1. Create a simple structured mesh
# -----------------------------------
# Generate a customizable structured BHE mesh:

# %%
tmp_dir = Path(mkdtemp())
vtu_file = tmp_dir / "bhe_structured.vtu"
bhe_meshes = gen_bhe_mesh(
    length=150,
    width=100,
    layer=[50, 50, 50],
    groundwater=Groundwater(
        begin=-30, isolation_layer_id=1, flow_direction="+x"
    ),
    BHE_Array=[
        BHE(x=50, y=40, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=50, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=60, z_begin=-1, z_end=-60, borehole_radius=0.076),
    ],
    meshing_type="structured",
    out_name=vtu_file,
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract the BHE lines:

# %%
mesh = pv.read(tmp_dir / bhe_meshes[0])
top_mesh = pv.read(tmp_dir / bhe_meshes[1])
bottom_mesh = pv.read(tmp_dir / bhe_meshes[2])
gw_mesh = pv.read(tmp_dir / bhe_meshes[3])
bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)

# %% [markdown]
# Visualize the mesh:

# %%
top_mesh = top_mesh.translate((0, 0, 10), inplace=False)
bottom_mesh = bottom_mesh.translate((0, 0, -10), inplace=False)
gw_mesh = gw_mesh.translate((-10, 0, 0), inplace=False)

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
p.add_axes()
p.show()

# %% [markdown]
# 2. Create an advanced structured mesh
# --------------------------------------
# Generate a customizable structured BHE mesh with advanced mesh
# sizing options (using gmsh). To understand the specific
# behaviour of every mesh parameter, test each one after another.

# %%
vtu_file = tmp_dir / "bhe_structured_advanced.vtu"
bhe_meshes = gen_bhe_mesh(
    length=150,
    width=100,
    layer=[50, 50, 50],
    groundwater=Groundwater(-30, 1, "+x"),
    BHE_Array=[
        BHE(50, 40, -1, -60, 0.076),
        BHE(50, 50, -1, -60, 0.076),
        BHE(50, 60, -1, -60, 0.076),
    ],
    meshing_type="structured",
    target_z_size_coarse=10,  # default value 7.5
    target_z_size_fine=2,  # default value 1.5
    n_refinement_layers=1,  # default value 2
    dist_box_x=15,  # default value 10
    propagation=1.2,  # default value 1.1
    inner_mesh_size=8,  # default value 5
    outer_mesh_size=12,  # default value 10
    out_name=vtu_file,
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract the BHE lines:

# %%
mesh = pv.read(tmp_dir / bhe_meshes[0])
top_mesh = pv.read(tmp_dir / bhe_meshes[1])
bottom_mesh = pv.read(tmp_dir / bhe_meshes[2])
gw_mesh = pv.read(tmp_dir / bhe_meshes[3])
bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)

# %% [markdown]
# Visualize the mesh:

# %%
top_mesh = top_mesh.translate((0, 0, 10), inplace=False)
bottom_mesh = bottom_mesh.translate((0, 0, -10), inplace=False)
gw_mesh = gw_mesh.translate((-10, 0, 0), inplace=False)

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
p.add_axes()
p.show()
