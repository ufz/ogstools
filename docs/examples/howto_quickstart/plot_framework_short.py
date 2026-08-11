"""
OGSTools Framework - Quick Start
=================================

This is a condensed version of the OGSTools workflow.

**Workflow:** Setup → Compose → Run → Analyze → Store
"""

# %% [markdown]
# For detailed explanations of each step, see :ref:`sphx_glr_auto_examples_howto_quickstart_plot_framework.py`.

# %%
import matplotlib.pyplot as plt

import ogstools as ot
from ogstools.examples import load_meshes_simple_lf, load_project_simple_lf

# %%
# 1. Setup: Load Project and Meshes
# ==================================

project = load_project_simple_lf()
meshes = load_meshes_simple_lf()

model = ot.Model(project=project, meshes=meshes)
# Visualize setup with boundary conditions
fig = model.plot_constraints()
plt.show()

# %%
# 2. Run: Execute Simulation
# ===========================

sim = model.run()
print(f"Simulation status: {sim.status_str}")

# %%
# 3. Analyze: Visualize Results
# ==============================

# Plot final pressure distribution
fig = ot.plot.contourf(sim.meshseries[-1], "pressure")
plt.show()


# %%
# Plot simulation time
df_ts = ot.logparser.analysis_time_step(sim.log.df_log).reset_index()
times = ["assembly_time", "dirichlet_time", "linear_solver_time"]
ax = df_ts.plot.area(x="time_step", y=times, ylabel="time / s", grid=True)
plt.show()

# %%
# 4. Store: Save Simulation
# ==========================

tmp = ot.definitions.temp_dir("framework_short", "examples")
sim.save(tmp / "mysim", archive=True)

# %% [markdown]
# Next Steps
# ==========
#
# - **Full tutorial**: :ref:`sphx_glr_auto_examples_howto_quickstart_plot_framework.py`
# - **Preprocessing**: :ref:`sphx_glr_auto_examples_howto_preprocessing_plot_gen_bhe_mesh.py`
# - **Visualization**: :ref:`sphx_glr_auto_examples_howto_plot_plot_timeslice.py`
# - **Storage**: :ref:`sphx_glr_auto_examples_howto_quickstart_plot_storage.py`
