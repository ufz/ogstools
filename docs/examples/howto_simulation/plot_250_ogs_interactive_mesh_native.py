"""
Interactive Mesh manipulation (Native interface)
================================================

This tutorial builds on the Interface OGS Simulation. If you haven't seen that, start there first.

Here we demonstrate how to access and modify in situ meshes during an OpenGeoSys simulation.

This allows you to:

- Inspect in situ mesh data

- Modify point/cell properties dynamically (e.g. boundary conditions)

- React to simulation progress and state

For a higher-level, more integrated approach, see the OGSTools-based examples.

This tutorial builds on the :ref:`sphx_glr_auto_examples_howto_simulation_plot_100_ogs_interactive_simulator.py`. For a lower-level control, see :ref:`sphx_glr_auto_examples_howto_simulation_plot_200_ogs_interactive_meshes_from_simulator.py`.

"""

# Imports and definitions
# =======================
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from ogs import OGSMesh
from ogs.OGSSimulator import OGSSimulation

import ogstools as ot
from ogstools.examples import load_model_liquid_flow_simple

# %%
# Select a Project
# ================
# We will start with a simple liquid flow project.

working_dir = Path(mkdtemp())
prj, meshes = load_model_liquid_flow_simple()
prj.write_input(working_dir / "LiquidFlowSimple.prj")
_ = meshes.save(working_dir)


# %%
# Read and write in situ meshes
# =============================
#
# We now start and connect to a running OGS simulation and access its meshes using the native Co-Simulation interface.
# For further details search for OGSSimulation in https://doxygen.opengeosys.org/search.html?query=OGSSimulation.


sim3_result_dir = working_dir / "sim3"
sim3_result_dir.mkdir(exist_ok=True)
arguments = [
    "",
    str(working_dir / "LiquidFlowSimple.prj"),
    "-o",
    str(sim3_result_dir),
]
sim3 = OGSSimulation(arguments)  # we will restart the same simulation as above

# %%
# Mesh Interface
# ==============

print("Available meshes:")
for name in sim3.mesh_names():
    print(name)

# Get any of the available meshes
left_boundary: OGSMesh = sim3.mesh("left")

print("Points:", left_boundary.points()[:10])
print("Cells:", left_boundary.cells()[:10])
print("Data array names:", left_boundary.data_array_names())
print("Mesh item type:", left_boundary.mesh_item_type("pressure"))

# Read an attribute
pressure = left_boundary.data_array("pressure", "double")

# %%
# Manipulation of values
# ======================
# We recommend to use numpy for manipulation of the values.
# Note: You can only modify the values, not the size of the array.

number_of_points = int(len(left_boundary.points()) / 3)  # always 3D in OGS

# Example to change a single value
pressure[0] = 3.1e7
# Example to change selected values
pressure[0:10] = np.linspace(1e7, 2e7, 9)
# Example to change all values
pressure[:] = np.full(number_of_points, 3.1e7, dtype=np.float32)  # Example 3

# Important: always assign directly to values
# Bad example (not changing the in situ mesh):
# pressure = np.full(mesh.n_points, 3.1e7, dtype=np.float32)

# %%
# Dynamic example
# ===============

for i in range(12):
    # Modify left boundary condition values
    if i < 6:
        # All points get the same pressure values
        pressure[:] = np.full(pressure.shape, 2.9e7)
    else:
        pressure[:] = np.full(pressure.shape, 3.1e7)

    # Now, with modified boundary conditions execute a single step
    # The step size is determined by setting in projectfile/timeloop
    # In this example we have fixed step
    sim3.execute_time_step()

# To run the simulation from last executed time step till the end (according to definitions in the project file)
sim3.execute_simulation()
sim3.close()

# For boundary conditions only (no interaction), see
# `Defining boundary conditions with Python <https://www.opengeosys.org/docs/userguide/features/python_bc/>`_.

# %%
# Visualization

ms3 = ot.MeshSeries(sim3_result_dir / "LiquidFlow_Simple.pvd")
# !paraview {ms.filepath} # for interactive exploration
# Time slice over x
points = np.linspace([0, 1, 0], [10, 1, 0], 100)
ms_probe = ot.MeshSeries.extract_probe(ms3, points, "pressure")
fig = ms_probe.plot_time_slice("time", "x", variable="pressure", num_levels=20)
