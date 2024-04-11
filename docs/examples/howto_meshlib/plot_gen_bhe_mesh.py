"""
Creating a BHE mesh (Borehole Heat Exchanger)
=============================================

This example demonstrates how to create a Borehole Heat Exchanger (BHE) mesh.
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import pyvista as pv
from pyvista.plotting import Plotter

from ogstools.meshlib.gmsh_meshing import gen_bhe_mesh
from ogstools.msh2vtu import msh2vtu

# %% [markdown]
# Generate a customizable BHE mesh (using gmsh):

# %%
tmp_dir = Path(mkdtemp())
msh_file = tmp_dir / "bhe.msh"
gen_bhe_mesh(
    length=150,
    width=100,
    layer=[50, 50, 50],
    BHE_array=[
        [50, 40, -1, -60, 0.076],
        [50, 50, -1, -60, 0.076],
        [50, 60, -1, -60, 0.076],
    ],
    groundwater=[[-30, 1, "+x"]],
    dist_box_x=5,
    meshing_type="structured",
    n_refinement_layers=1,
    outer_mesh_size=12,
    inner_mesh_size=5,
    propagation=1.2,
    out_name=msh_file,
)

# %% [markdown]
# Now we convert the gmsh mesh to the VTU format with msh2vtu.
# Passing the list of dimensions [1, 3] to msh2vtu ensures, that the line
# elements will also be part of the domain mesh.

# %%
msh2vtu(
    msh_file, output_path=tmp_dir, dim=[1, 3], reindex=True, log_level="ERROR"
)

# %% [markdown]
# Load the domain mesh and extract BHE line:

# %%
mesh = pv.read(tmp_dir / "bhe_domain.vtu")
bhe_line = mesh.extract_cells_by_type(pv.CellType.LINE)

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
p.show()
