"""
OGSTools Framework - Quick Start
=================================

This is a condensed version of the OGSTools workflow.

**Workflow:** Setup → Compose → Run → Analyze → Store
"""

# %% [markdown]
# For detailed explanations of each step, see :ref:`sphx_glr_auto_examples_howto_quickstart_plot_framework.py`.

# %%
import tempfile
from pathlib import Path

import numpy as np

import ogstools as ot
from ogstools.examples import (
    load_meshes_simple_lf,
    load_project_simple_lf,
)

# %%
# 1. Setup: Load Project and Meshes
# ==================================

project = load_project_simple_lf()
meshes = load_meshes_simple_lf()

# Set initial conditions
meshes["right"].point_data["pressure"] = np.full(
    np.shape(meshes["left"].points)[0], 3.1e7
)

# Visualize mesh topology
fig = meshes.plot()

model = ot.Model(project=project, meshes=meshes)
# Visualize setup with boundary conditions
fig = model.plot_constraints()

# %%
# 2. Run: Execute Simulation
# ===========================

sim = model.run()
print(f"Simulation status: {sim.status_str}")

# %%
# 3. Analyze: Visualize Results
# ==============================

# Plot final pressure distribution
ot.plot.contourf(sim.meshseries[-1], "pressure")

# Plot convergence behavior
sim.log.plot_convergence()

# %%
# 4. Store: Save Simulation
# ==========================

tmp = Path(tempfile.mkdtemp())
sim.save(tmp / "mysim", archive=True)

# %% [markdown]
# Next Steps
# ==========
#
# - **Full tutorial**: :ref:`sphx_glr_auto_examples_howto_quickstart_plot_framework.py`
# - **Preprocessing**: :ref:`sphx_glr_auto_examples_howto_preprocessing_plot_gen_bhe_mesh.py`
# - **Visualization**: :ref:`sphx_glr_auto_examples_howto_plot_plot_timeslice.py`
# - **Storage**: :ref:`sphx_glr_auto_examples_howto_quickstart_plot_storage.py`
