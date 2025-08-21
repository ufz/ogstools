"""
Interactive OpenGeoSys execution (Co-Simulation)
================================================

Interactive execution. With interactive execution, OpenGeoSys provides a Python interface that lets you
advance a simulation one step at a time, inspect intermediate results, and react before continuing.
This mechanism underlies what we mean by Co-simulation: the simulation can exchange information with
external analyses or tools during runtime—for example, by dynamically adjusting boundary conditions.
In this guide, we will show how these interactions can be carried out using OGSTools.

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

mesh_path = Path(mkdtemp())
prj_path = EXAMPLES_DIR / "prj" / "SimpleLF.prj"
gmsh_mesh_name = mesh_path / "rect.msh"


# %%
# 1. Create mesh and boundary meshes
# ==================================
#
# For this example we create the domain mesh (a 10x2 rectangle) and the boundary meshes for the simulation
# Furthermore, the boundary
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
# Save the domain and boundary meshes (in gmsh format)
for name, sub_mesh in meshes.items():
    pv.save_meshio(Path(mesh_path, name + ".vtu"), sub_mesh)


# %%
# 2. Initialize the simulation and first step
# ===========================================
# Use same arguments as when calling ogs from command line, --> link
# https://www.opengeosys.org/docs/userguide/basics/cli-arguments/

results_path = Path(mkdtemp())
arguments = [
    "",
    str(prj_path),
    "-l debug",  # optional
    "-m",  # optional, if mesh is located on different folder then prj file
    str(mesh_path),
    "-o",
    str(results_path),
]

# Initialize starts OpenGeoSys and runs time step 0
initialization_status = simulator.initialize(arguments)

# %%
print("The current time step is:", simulator.currentTime())

# In the results_path output files for this time step have been created:
for file in results_path.iterdir():
    print(file.name)


# %%
# 3. Single step
status = simulator.executeTimeStep()

# Time stepping has advanced
print("The current time step is:", simulator.currentTime())


# %%
# 3.1 Interaction with OGSTools
# =============================

# Let us interact with the domain and the left boundary mesh
domain_mesh = ot.Mesh.from_simulator(simulator, "domain")
left_mesh = ot.Mesh.from_simulator(simulator, "physical_group_left")

# For all points of the left boundary mesh set the pressure value
# We recommend to use numpy for manipulation of the values.
# Important: You can only change the values, not the shape / length of the array.
left_mesh.point_data["pressure"] = np.full(
    np.shape(left_mesh.number_of_points), 1e8
)


# %%
# 3.2 Interaction with OpenGeoSys interface
# ========================================
# For performance critical applications you can skip the conversion to pyvista and directly
# use the Co-Simulation interface of OpenGeoSys.
# Link to complete API https://doxygen.opengeosys.org/d1/d7b/classsimulation

# If the initialization was successful you can access to the current state of
# the mesh via the getMesh() method
# https://doxygen.opengeosys.org/d9/de9/classogsmesh

# getMesh
left_boundary: mesh = simulator.getMesh("physical_group_left")


pressure = np.array(left_boundary.getPointDataArray("pressure", 1))
# We recommend to use numpy for manipulation of the values.
# Important: You can only change the values, not the shape / length of the array.

# %%
# Replace property values of the mesh with setCellDataArray or setPointDataArray


for i in range(12):
    # Modify left boundary condition values
    if i < 6:
        # All points get the same pressure values
        pressure = np.full(pressure.shape, 2.99e7)
    else:
        pressure = np.full(pressure.shape, 3.01e7)

    # Write values back to the simulator (OpenGeoSys)
    left_boundary.setPointDataArray("pressure", pressure, 1)
    # Use setCellDataArray for node-centred properties

    # Now, with modified boundary conditions execute a single step
    # The step size is determined by setting in projectfile/timeloop
    # In this example we have fixed step
    simulator.executeTimeStep()


# %%
# 4. Finalize the simulation
# To run the simulation till the end (according to definitions in the project file)

simulator.executeSimulation()


# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments)
# %%
simulator.finalize()

# %%
