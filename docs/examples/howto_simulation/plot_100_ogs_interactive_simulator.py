"""
Run a simulation - Step by step
===============================

With interactive execution, OpenGeoSys provides a Python interface that lets you
advance a simulation one step at a time, inspect intermediate results, and react before continuing.
This mechanism underlies what we mean by Co-simulation: the simulation can exchange information with
external analyses or tools during runtime—for example, by dynamically adjusting boundary conditions.
In this guide, we will show how to:

- Step-by-step simulation control
- Inspect simulation status and intermediate results
- Prepare for mesh manipulation examples
  - :ref:`sphx_glr_auto_examples_howto_simulation_plot_200_ogs_interactive_meshes_from_simulator.py`.
  - :ref:`sphx_glr_auto_examples_howto_simulation_plot_250_ogs_interactive_mesh_native.py`.




"""

# %%
# Imports and definitions
# =======================
from pathlib import Path
from tempfile import mkdtemp

from ogs import OGSMesh
from ogs.OGSSimulator import OGSSimulation

from ogstools.examples import load_model_liquid_flow_simple

# %%
# Select a Project
# ================
# We will start with a simple liquid flow example.

working_dir = Path(mkdtemp())
prj, meshes = load_model_liquid_flow_simple()
prj.write_input(working_dir / "LiquidFlowSimple.prj")
_ = meshes.save(working_dir)


# %%
# Initialize the simulation and first step
# ========================================
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
# By constructing a OGSSimulation object we start OpenGeoSys and it runs time step 0
sim1 = OGSSimulation(arguments)

# %%
print(f"The current simulation time is: {sim1.current_time()} s.")
print(f"The end time defined is: {sim1.end_time()} s.")


# %%
# Advance the simulation by a one step
# ====================================
status = sim1.execute_time_step()
print(f"The current simulation time is: {sim1.current_time()} s.")

# %%
# Basic simulation loop
# =====================
#
# The simulation runs step-by-step until a user specified condition is met.
# This further allows you to **manipulate data arrays associated with the mesh or adapt the logic** inside the loop.

while (
    # Here, the condition is when simulation reaches the end time:
    sim1.current_time()
    < sim1.end_time()
):

    # Add here your specific code with mesh manipulation:
    my_mesh: OGSMesh = sim1.mesh("domain")
    sim1.execute_time_step()

# Alternatively, run the entire simulation without interaction:
sim1.execute_simulation()
# Necessary to close, otherwise you can not reinitialize simulation with same prj-file (arguments):
sim1.close()


# %%
# Advance examples with mesh manipulation
# =======================================
#
# Manipulate the mesh with either
#
# - :ref:`sphx_glr_auto_examples_howto_simulation_plot_200_ogs_interactive_meshes_from_simulator.py`.
# - :ref:`sphx_glr_auto_examples_howto_simulation_plot_250_ogs_interactive_mesh_native.py`.
