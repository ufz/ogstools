"""
Interactive OpenGeoSys execution (Co-Simulation)
================================================

With interactive execution, OpenGeoSys provides a Python interface that lets you
advance a simulation one step at a time, inspect intermediate results, and react before continuing.
This mechanism underlies what we mean by Co-simulation: the simulation can exchange information with
external analyses or tools during runtime—for example, by dynamically adjusting boundary conditions.
In this guide, we will show how these interactions can be carried out.

Alternatively, for boundary conditions only (no interaction), have a look at our guide about
`Defining boundary conditions with Python <https://www.opengeosys.org/docs/userguide/features/python_bc/>`_.


"""

# %%
# Imports and definitions
from pathlib import Path
from tempfile import mkdtemp
from time import sleep  # For simulation pause / interrupt

import numpy as np
import pyvista as pv
from ogs import mesh, simulator  # Python API to a running OGS

import ogstools as ot
from ogstools._find_ogs import interrupted as current_sim_interrupted
from ogstools.definitions import EXAMPLES_DIR

mesh_path = Path(mkdtemp())
prj_path = EXAMPLES_DIR / "prj" / "SimpleLF.prj"
gmsh_mesh_name = mesh_path / "rect.msh"


# %%
# 1. Create mesh and boundary meshes
# ==================================
#
# In this example, we create the domain mesh (a 10x2 rectangle) and the boundary meshes for the simulation
# We also set the boundary condition (prescribed pressure) on the left and right boundary meshes.
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
    points_shape[0], 2.9e7
)
meshes["physical_group_right"].point_data["pressure"] = np.full(
    points_shape[0], 3e7
)

# Save the domain and boundary meshes (in gmsh format)
for name, sub_mesh in meshes.items():
    pv.save_meshio(Path(mesh_path, name + ".vtu"), sub_mesh)


# %%
# 2. Initialize the simulation and first step
# ===========================================
# Use the same arguments as when calling OGS from command line, a list of
# possible argument are documented under
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
print(f"The current simulation time is: {simulator.currentTime()} s.")
print(f"The end time defined is: {simulator.endTime()} s.")

# %%
# These output files for this time step are created in results_path:
for file in results_path.iterdir():
    print(file.name)


# %%
# 3. Advancing the simulation by a single step
# ============================================
# Advance a single step and read the time of the current step.
status = simulator.executeTimeStep()
print(f"The current simulation time is: {simulator.currentTime()} s.")

# Now let us put this into a simple loop that runs until the end time is reached.

# %%
# 4. Basic simulation loop
# ========================
#
# The following loop runs the simulation step-by-step until the end time is reached (or the user interrupts it).
while (
    # Condition to Stop when simulation reaches the end time
    simulator.currentTime()  # in s, value changes with invoking executeTimeStep
    < simulator.endTime()  # in s, value stays constant (defined in prj file - timeloop)
    and not current_sim_interrupted()  # Stops if the user presses Ctrl+C or interrupts the notebook.
):

    simulator.executeTimeStep()
    sleep(0.01)  # This is required if you want to pause the simulation.
    # Here you may add custom code to interact with data of current time step
    # See section 5

# Or just run without any interaction from current time step till the end of the simulation
simulator.executeSimulation()
# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments)
simulator.finalize()

# Next, we build a more advanced loop that adapts boundary conditions dynamically.

# %%
# 5. Co-simulation with the native OGS interface
# ==============================================
# This example shows how to run an OGS simulation interactively,
# monitor convergence (steady state), and dynamically change boundary conditions during the run.
# Use it as a template for adaptive simulation loops.
# To get the current mesh from a running OGS simulation, use
# :py:meth:`ot.Mesh.create_from_simulator <ogstools.meshlib.Mesh.update_from_simulator>`
# You can read / manipulate this mesh using pyvista functionality.
# Send this pyvista mesh back to the simulator with :py:meth:`~ogstools.meshlib.Mesh.write_to_simulator`.
# Use :py:meth:`~ogstools.meshlib.Mesh.update_from_simulator` as a performance tuned method, when you already have received the mesh.

simulator.initialize(arguments)  # we will restart the same simulation as above
domain_mesh = ot.Mesh.from_simulator(simulator, "domain", ["pressure"])

# Let us store the mesh of a time step to compare it later
previous_domain_mesh_pressure = domain_mesh.point_data["pressure"]
domain_mesh_pressure = previous_domain_mesh_pressure

# Later, we will dynamically modify the left boundary mesh
left_mesh = ot.Mesh.from_simulator(
    simulator, "physical_group_left", ["pressure"]
)

steady_state_threshold = 0.1 * len(previous_domain_mesh_pressure)  # constant
delta = steady_state_threshold + 1e-10  # to be computed for each time step

# %%
# Main simulation loop running till defined simulation end
# OR interrupted by user
while (
    simulator.currentTime()
    < simulator.endTime()  # Stop when simulation reaches the end time
    and not current_sim_interrupted()  # Stop if user presses Ctrl-C or SIGTERM is received (e.g. Stop Button in Jupyter Notebook)
    # any other stopping condition based on the mesh, log, ...
    # e.g. # `delta > steady_state_threshold``
):

    previous_domain_mesh_pressure = domain_mesh.point_data["pressure"]  # Copy

    simulator.executeTimeStep()
    # Must have, if you want to pause the simulation
    sleep(0.01)  # directly after `executeTimeStep`` is often a good place

    # Example for an In-loop condition
    # 1. Check, if a steady state is reached
    # 2. Change a boundary condition
    domain_mesh.update_from_simulator(simulator)
    delta = np.max(
        np.absolute(
            domain_mesh.point_data["pressure"] - previous_domain_mesh_pressure
        )
    )
    if delta < steady_state_threshold:
        print(
            "The steady state condition is reached. Now a new boundary condition is applied."
        )
        left_mesh.point_data["pressure"] = np.full(
            np.shape(left_mesh.number_of_points), 3.1e7
        )
        left_mesh.write_to_simulator(simulator)
    # else: we keep the boundary conditions constant

# This simulation reaches the steady state condition twice
# 1st with left boundary pressure at 2.9e7 Pa, 2nd: 3.1e7 Pa

# %%
# If user-stopped(Ctrl+C or Jupytercell-Interrupt), you may now investigate, adapt and continue the simulation as shown in while loop above
# e.g. current pressure
fig = ot.plot.contourf(domain_mesh, "pressure")

# %%
# Continue the ("paused") simulation with a single step
simulator.executeTimeStep()
# Or run over multiple steps (see while loop above)
simulator.executeSimulation()
# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments)
simulator.finalize()

# %%
# The influence of the changed boundary condition can be observed by the significant increase in
# the pressure value after half of the simulation time.
# The left boundary (x=0) changes after 60 s
# The right boundary (x=10) changes after the first time step
ms = ot.MeshSeries(results_path / "LiquidFlow_Simple.pvd")
# !paraview {ms.filepath} # for interactive exploration
# Time slice over x
points = np.linspace([0, 1, 0], [10, 1, 0], 100)
ms_probe = ot.MeshSeries.extract_probe(ms, points, "pressure")
fig = ms_probe.plot_time_slice("time", "x", variable="pressure", num_levels=20)

# %%
# 7. Advanced Interaction with the OpenGeoSys interface
# =====================================================
# For performance-critical applications, you can bypass the conversion to pyvista
# and use the Co-Simulation interface of OpenGeoSys directly.
# use the Co-Simulation interface of OpenGeoSys.
# Link to complete API https://doxygen.opengeosys.org/d1/d7b/classsimulation

# If the initialization was successful you can access the current state of
# the mesh via the getMesh() method
# https://doxygen.opengeosys.org/d9/de9/classogsmesh

simulator.initialize(arguments)  # we will restart the same simulation as above

# getMesh
left_boundary: mesh = simulator.getMesh("physical_group_left")


pressure = np.array(left_boundary.getPointDataArray("pressure", 1))
# We recommend to use numpy for manipulation of the values.
# Note: You can only modify the values, not the size of the array.

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

# To run the simulation from last executed time step till the end (according to definitions in the project file)
simulator.executeSimulation()

# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments)
simulator.finalize()
