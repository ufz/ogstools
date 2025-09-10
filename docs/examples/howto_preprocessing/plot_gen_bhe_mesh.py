# %%
"""
Creating a BHE mesh (Borehole Heat Exchanger)
===================================================
"""

# %% [markdown]
# This example demonstrates how to use :py:mod:`ogstools.meshlib.gmsh_BHE.gen_bhe_mesh` to create
# a Borehole Heat Exchanger (BHE) mesh.

# %%
from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv
from shapely import Polygon

import ogstools as ot
from ogstools.meshlib.gmsh_BHE import BHE, Groundwater, gen_bhe_mesh

# %% [markdown]
# Introduction
# ----------------
# This example shows the general usage of :py:mod:`ogstools.meshlib.gmsh_BHE.gen_bhe_mesh` and how some of
# the parameters will effect the mesh. This section demonstrates the mesh
# generation with only three soil layers, groundwater flow in one layer
# and three BHE's. However, this tool provides multiple soil layers,
# groundwater flow in multiple layers and multiple BHE's. The mesh sizes
# provides good initial values for the most Heat-Transport-BHE simulations
# in OGS. They can also be set by the user, to customize the mesh.
# Feel free to try it out!

# %% [markdown]
# Create a simple prism mesh
# --------------------------------
# Generate a customizable prism BHE mesh:

# %%
tmp_dir = Path(mkdtemp())
vtu_file = tmp_dir / "bhe_prism.vtu"
bhe_meshes = gen_bhe_mesh(
    model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=150, ymax=100),
    layer=[50, 50, 50],
    groundwater=Groundwater(
        begin=-30,
        isolation_layer_id=1,
        upstream=(179, 181),
        downstream=(359, 1),
    ),
    BHE_Array=[
        BHE(x=50, y=40, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=50, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=60, z_begin=-1, z_end=-60, borehole_radius=0.076),
    ],
    refinement_area=Polygon.from_bounds(xmin=40, ymin=30, xmax=60, ymax=70),
    meshing_type="prism",
    out_name=vtu_file,
)

# %% [markdown]
# Load the domain mesh and all submeshes, extract the BHE lines and visualize it all:


# %%
def load_and_plot(bhe_meshes: list[Path]):
    mesh: pv.UnstructuredGrid = pv.read(tmp_dir / bhe_meshes[0])
    bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)
    top_bottom_gw_mesh = [
        pv.read(tmp_dir / bhe_meshes[idx]) for idx in range(1, 4)
    ]
    offsets = [(0, 0, 10), (0, 0, -10), (-10, 0, 0)]
    plotter = ot.plot.contourf(
        mesh.clip("x", bhe_line.center, crinkle=True), ot.variables.material_id
    )
    plotter.add_mesh(mesh, style="wireframe", color="grey")
    plotter.add_mesh(bhe_line, color="r", line_width=3)
    for submesh, offset in zip(top_bottom_gw_mesh, offsets, strict=True):
        plotter.add_mesh(submesh.translate(offset), show_edges=True)
    plotter.show()


# %%

load_and_plot(bhe_meshes)

# %% [markdown]
# Create a simple structured mesh
# -----------------------------------
# Generate a customizable structured BHE mesh:

# %%
tmp_dir = Path(mkdtemp())
vtu_file = tmp_dir / "bhe_structured.vtu"
bhe_meshes = gen_bhe_mesh(
    model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=150, ymax=100),
    layer=[50, 50, 50],
    groundwater=Groundwater(
        begin=-30,
        isolation_layer_id=1,
        upstream=(179, 181),
        downstream=(359, 1),
    ),
    BHE_Array=[
        BHE(x=50, y=40, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=50, z_begin=-1, z_end=-60, borehole_radius=0.076),
        BHE(x=50, y=60, z_begin=-1, z_end=-60, borehole_radius=0.076),
    ],
    refinement_area=Polygon.from_bounds(xmin=40, ymin=30, xmax=60, ymax=70),
    meshing_type="structured",
    out_name=vtu_file,
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract the BHE lines:

# %%
load_and_plot(bhe_meshes)

# %% [markdown]
# Create an advanced structured mesh
# --------------------------------------
# Generate a customizable structured BHE mesh with advanced mesh
# sizing options (using gmsh). By adding additional points to the model_area,
# the subdivison of the surface by the structured mesh algorithm is controlled.
# To understand the specific behaviour of every mesh parameter,
# test each one after another.
#
# .. image:: ../../examples/howto_preprocessing/gen_bhe_mesh.svg


# %%
vtu_file = tmp_dir / "bhe_structured_advanced.vtu"
bhe_meshes = gen_bhe_mesh(
    # add additional points for better subsidivision of the surface by structured algorithm
    # compare with previous example, to see the difference
    model_area=Polygon(
        (
            (0.0, 0.0),
            (40.0, 0.0),
            (60.0, 0.0),
            (150.0, 0.0),
            (150.0, 100.0),
            (60.0, 100.0),
            (40.0, 100.0),
            (0.0, 100.0),
        )
    ),
    layer=[50, 50, 50],
    groundwater=Groundwater(
        begin=-30,
        isolation_layer_id=1,
        upstream=(179, 181),
        downstream=(359, 1),
    ),
    BHE_Array=[
        BHE(50, 40, -1, -60, 0.076),
        BHE(50, 50, -1, -60, 0.076),
        BHE(50, 60, -1, -60, 0.076),
    ],
    refinement_area=Polygon.from_bounds(xmin=40, ymin=30, xmax=60, ymax=70),
    meshing_type="structured",
    target_z_size_coarse=10,  # default value 7.5
    target_z_size_fine=2,  # default value 1.5
    n_refinement_layers=1,  # default value 2
    propagation=1.2,  # default value 1.1
    inner_mesh_size=8,  # default value 5
    outer_mesh_size=12,  # default value 10
    out_name=vtu_file,
)

# %% [markdown]
# Load the domain mesh and all submeshes as well as extract the BHE lines:

# %%
load_and_plot(bhe_meshes)
