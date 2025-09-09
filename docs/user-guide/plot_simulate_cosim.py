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
from ogs import OGSSimulator, mesh  # Python API to a running OGS

import ogstools as ot
from ogstools._find_ogs import interrupted as current_sim_interrupted
from ogstools.examples import load_model_liquid_flow_simple

# %%
# 1. Select a Project
# ===================

# Here liquid flow
working_dir = Path(mkdtemp())
prj, meshes = load_model_liquid_flow_simple()
prj.write_input(working_dir / "LiquidFlowSimple.prj")
_ = meshes.save(working_dir)

# model.run(interactive=True)
# model.run(dry=True)


# %%
# 2. Initialize the simulation and first step
# ===========================================
# Use the same arguments as when calling OGS from command line, a list of
# possible argument are documented under
# https://www.opengeosys.org/docs/userguide/basics/cli-arguments/

sim1_result_dir = working_dir / "sim1"
sim1_result_dir.mkdir(exist_ok=True)
arguments = [
    "",
    str(working_dir / "LiquidFlowSimple.prj"),
    "-o",
    str(sim1_result_dir),
]
# extend like this: ,"-m", str(mesh_dir), "-l debug", -o", str(output_dir)]

# %%
# Initialize starts OpenGeoSys and runs time step 0

sim1 = OGSSimulator.OGSSimulation(arguments)


# %%
print(f"The current simulation time is: {sim1.currentTime()} s.")
print(f"The end time defined is: {sim1.endTime()} s.")

# %%
# These output files for this time step are created in results_path:
for file in working_dir.iterdir():
    print(file.name)


# %%
# 3. Advancing the simulation by a single step
# ============================================
# Advance a single step and read the time of the current step.
status = sim1.executeTimeStep()
print(f"The current simulation time is: {sim1.currentTime()} s.")

# %%
# 4. Basic simulation loop
# ========================
#
# The following loop runs the simulation step-by-step until the end time is reached (or the user interrupts it).
while (
    # Condition to Stop when simulation reaches the end time
    sim1.currentTime()  # in s, value changes with invoking executeTimeStep
    < sim1.endTime()  # in s, value stays constant (defined in prj file - timeloop)
    and not current_sim_interrupted()  # Stops if the user presses Ctrl+C or interrupts the notebook.
):

    sim1.executeTimeStep()
    sleep(0.01)  # This is required if you want to pause the simulation.
    # Here you may add custom code to interact with data of current time step
    # See section 5

# Or just run without any interaction from current time step till the end of the simulation
sim1.executeSimulation()
# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments)
sim1.finalize()

# Next, we build a more advanced loop that adapts boundary conditions dynamically.

# %%
# 5. Co-simulation with the native OGS interface - Using OGSTools
# ===============================================================
# This example shows how to run an OGS simulation interactively,
# monitor convergence (steady state), and dynamically change boundary conditions during the run.
# Use it as a template for adaptive simulation loops.
# To get the current mesh from a running OGS simulation, use
# :py:meth:`ot.Mesh.create_from_simulator <ogstools.meshlib.Mesh.update_from_simulator>`
# You can read / manipulate this mesh using pyvista functionality.
# Send this pyvista mesh back to the simulator with :py:meth:`~ogstools.meshlib.Mesh.write_to_simulator`.
# Use :py:meth:`~ogstools.meshlib.Mesh.update_from_simulator` as a performance tuned method, when you already have received the mesh.

sim2_result_dir = working_dir / "sim2"
sim2_result_dir.mkdir(exist_ok=True)
arguments = [
    "",
    str(working_dir / "LiquidFlowSimple.prj"),
    "-o",
    str(sim2_result_dir),
]
sim2 = OGSSimulator.OGSSimulation(
    arguments
)  # we will restart the same simulation as above into new folder
domain_mesh = ot.Mesh.from_simulator(sim2, "domain", ["pressure"])

# Let us store the mesh of a time step to compare it later
previous_domain_mesh_pressure = domain_mesh.point_data["pressure"]
domain_mesh_pressure = previous_domain_mesh_pressure

# Later, we will dynamically modify the left boundary mesh
left_mesh = ot.Mesh.from_simulator(sim2, "physical_group_left", ["pressure"])

steady_state_threshold = 0.1 * len(previous_domain_mesh_pressure)  # constant
delta = steady_state_threshold + 1e-10  # to be computed for each time step

# %%
# Main simulation loop running till defined simulation end
# OR interrupted by user
while (
    sim2.currentTime()
    < sim2.endTime()  # Stop when simulation reaches the end time
    and not current_sim_interrupted()  # Stop if user presses Ctrl-C or SIGTERM is received (e.g. Stop Button in Jupyter Notebook)
    # any other stopping condition based on the mesh, log, ...
    # e.g. # `delta > steady_state_threshold``
):

    previous_domain_mesh_pressure = domain_mesh.point_data["pressure"].copy()

    sim2.executeTimeStep()
    # Must have, if you want to pause the simulation
    sleep(0.01)  # directly after `executeTimeStep`` is often a good place

    # Example for an In-loop condition
    # 1. Check, if a steady state is reached
    # 2. Change a boundary condition
    # domain_mesh.update_from_simulator(sim2)
    delta = np.max(
        np.absolute(
            domain_mesh.point_data["pressure"] - previous_domain_mesh_pressure
        )
    )
    if delta < steady_state_threshold:
        print(
            "The steady state condition is reached. Now a new boundary condition is applied."
        )
        # [:] Is important! Otherwise you would assign a new numpy array to pressure (which would write into the insitu mesh)
        left_mesh.point_data["pressure"][:] = np.full(left_mesh.n_points, 3.1e7)
    # else: we keep the boundary conditions constant

# This simulation reaches the steady state condition twice
# 1st with left boundary pressure at 2.9e7 Pa, 2nd: 3.1e7 Pa

# %%
# If user-stopped(Ctrl+C or Jupytercell-Interrupt), you may now investigate, adapt and continue the simulation as shown in while loop above
# e.g. current pressure
fig = ot.plot.contourf(domain_mesh, "pressure")

# %%
# Continue the ("paused") simulation with a single step
sim2.executeTimeStep()
# Or run over multiple steps (see while loop above)
sim2.executeSimulation()
# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments)
sim2.finalize()


# %%
# The influence of the changed boundary condition can be observed by the significant increase in
# the pressure value after half of the simulation time.
# The left boundary (x=0) changes after 60 s
# The right boundary (x=10) changes after the first time step
ms = ot.MeshSeries(sim2_result_dir / "LiquidFlow_Simple.pvd")
# !paraview {ms.filepath} # for interactive exploration
# Time slice over x
points = np.linspace([0, 1, 0], [10, 1, 0], 100)
ms_probe = ot.MeshSeries.extract_probe(ms, points, "pressure")
fig = ms_probe.plot_time_slice("time", "x", variable="pressure", num_levels=20)

# %%
# 6. Advanced Interaction with the OpenGeoSys interface
# =====================================================
# For performance-critical applications, you can bypass the conversion to pyvista
# and use the Co-Simulation interface of OpenGeoSys directly.
# use the Co-Simulation interface of OpenGeoSys.
# Link to complete API https://doxygen.opengeosys.org/d1/d7b/classsimulation

# If the initialization was successful you can access the current state of
# the mesh via the getMesh() method
# https://doxygen.opengeosys.org/d9/de9/classogsmesh


sim3_result_dir = working_dir / "sim3"
sim3_result_dir.mkdir(exist_ok=True)
arguments = [
    "",
    str(working_dir / "LiquidFlowSimple.prj"),
    "-o",
    str(sim3_result_dir),
]
sim3 = OGSSimulator.OGSSimulation(
    arguments
)  # we will restart the same simulation as above

# %%
# %% 7.1 Simulator Interface

# Control
print("CurrentTime:", sim3.currentTime())
sim3.executeTimeStep()
# sim3.executeSimulation() runs till defined simulation end
# sim3.finalize() to close simulation

# Exploration
print("Available meshes:")
for name in sim3.getMeshNames():
    print(name)

# Get any of the available meshes
left_boundary: mesh = sim3.getMesh("physical_group_left")

# %% 7.2 Native Mesh Interface
# explore the Mesh
print("Some points:", left_boundary.getPointCoordinates()[:10])
print("Some cells:", left_boundary.getCells()[:10])
print("Data array names:", left_boundary.dataArrayNames())
print("Mesh item type:", left_boundary.meshItemType("pressure"))

# Read an attribute
pressure = left_boundary.dataArray("pressure", "double")

# %%
# We recommend to use numpy for manipulation of the values.
# Note: You can only modify the values, not the size of the array.


pressure[0] = 3.1e7  # Example 1
number_of_points = int(
    len(left_boundary.getPointCoordinates()) / 3
)  # OGS has only 3D points
pressure[0:10] = np.linspace(1e7, 2e7, 9)  # Example 2
pressure[:] = np.full(number_of_points, 3.1e7, dtype=np.float32)  # Example 3
# Important: always assign directly to values
# Not pressure = np.full(mesh.n_points, 3.1e7, dtype=np.float32)

# %%
# 7.3 Dynamic example

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
    sim3.executeTimeStep()

# To run the simulation from last executed time step till the end (according to definitions in the project file)
sim3.executeSimulation()
sim3.finalize()

# %%

ms3 = ot.MeshSeries(sim3_result_dir / "LiquidFlow_Simple.pvd")
# !paraview {ms.filepath} # for interactive exploration
# Time slice over x
points = np.linspace([0, 1, 0], [10, 1, 0], 100)
ms_probe = ot.MeshSeries.extract_probe(ms3, points, "pressure")
fig = ms_probe.plot_time_slice("time", "x", variable="pressure", num_levels=20)
# %%
