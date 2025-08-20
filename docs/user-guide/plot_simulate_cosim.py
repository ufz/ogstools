"""
Co-Simulation
=============

OpenGeoSys offers a python interface that can be used to control some aspects of
a simulation from outside. For example, using the python interface it is
possible to 'step' through the simulation and analyse the intermediate results
of a particular step. Based on the analysis, for instance, the boundary
conditions can be adjusted.

"""

# %%
# Imports and definitions
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyvista as pv
from ogs import mesh, simulator

import ogstools as ot
from ogstools.definitions import EXAMPLES_DIR

results_path = Path(mkdtemp())
prj_path = EXAMPLES_DIR / "prj" / "SimpleLF.prj"
gmsh_mesh_name = results_path / "rect.msh"


# %%
# Create mesh and boundary meshes
#
# The domain mesh (a 10x2 rectangle) and the boundary meshes for the simulation
# will be created in the following code section. Furthermore, the boundary
# condition (prescribed pressure) is set on the left and right boundary meshes.

ot.meshlib.rect(
    lengths=(10, 2),
    n_edge_cells=(10, 4),
    n_layers=2,
    structured_grid=True,
    order=1,
    mixed_elements=False,
    jiggle=0.0,
    out_name=Path(gmsh_mesh_name),
)

meshes = ot.meshes_from_gmsh(gmsh_mesh_name)

# %%
# Add data array 'pressure' to the left and right meshes boundary meshes

points_shape = np.shape(meshes["physical_group_left"].points)
meshes["physical_group_left"].point_data["pressure"] = np.full(
    points_shape[0], 2.99e7
)
meshes["physical_group_right"].point_data["pressure"] = np.full(
    points_shape[0], 3e7
)

# %%
# save the domain and boundary meshes (in gmsh format)
for name, sub_mesh in meshes.items():
    pv.save_meshio(Path(results_path, name + ".vtu"), sub_mesh)


# %%
# Initialize the simulation
# Use same arguments as when calling ogs from command line, --> link

arguments = [
    "",
    str(prj_path),
    "-l debug",  # optional
    "-m",  # optional, if mesh is located on different folder then prj file
    str(results_path),
    "-o",
    str(results_path),
]

simulator.initialize(arguments)

# %%
# Get access to the current state of the mesh (here left boundary mesh)
# link to getMesh

left_boundary: mesh = simulator.getMesh("physical_group_left")
pressure = np.array(left_boundary.getPointDataArray("pressure", 1))


# plot pressure

# todo extract number of timesteps from project file


# %%
# Replace property values of the mesh with setCellDataArray or setPointDataArray
#

for i in range(15):
    # modify left boundary condition values
    if i < 10:
        pressure = np.full(pressure.shape, 2.99e7)
    else:
        pressure = np.full(pressure.shape, 3.01e7)

    # or setPointDataArray for node-centred properties
    left_boundary.setCellDataArray("pressure", pressure, 1)

    # Perform a single step (step size is determined by setting in projectfile/timeloop)
    # In this example we have fixed step
    simulator.executeTimeStep()

    ## ToDo getTimeStep?


# %%

# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments)

simulator.finalize()


# %%

# Link to complete API https://doxygen.opengeosys.org/d1/d7b/classsimulation
