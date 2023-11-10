"""
Convergence study (temporal refinement)
=======================================

This example shows one possible implementation of how to do a convergence study
with temporal refinement. For this, a simple model using a time dependent heat
source on one side and constant temperature on the opposite side was set up.
The heat source is generated with the `nuclearwasteheat` model.

Here is some theoretical background for the topic of grid convergence:
https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html
https://curiosityfluids.com/2016/09/09/establishing-grid-convergence/

At least three meshes from simulations of increasing temporal refinement are
required for the convergence study. The topology has to stay the same.

The below code cells will generate the simulation results and are evaluated for
convergence at the end.

First, the required packages are imported and an output directory is created:
"""

# %%
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import matplotlib.pyplot as plt
import numpy as np
import vtuIO
from IPython.display import HTML
from ogs6py import ogs

from ogstools import meshlib, msh2vtu, physics, propertylib, studies, workflow

temp_dir = Path(mkdtemp(prefix="nuclear_decay"))

# %% [markdown]
# Let's Visualize the temporal evolution of the source term and it's
# discretization in the simulations. We see, that with coarse time steps, the
# applied heat will be overestimated at first and underestimated once the heat
# curve has reached its maximum. The same is true for the resulting temperature.

# %%
n_refinements = 4
t_end = 180.0
sec_per_yr = 365.25 * 86400.0
time = np.append(0.0, np.geomspace(1.0, t_end, num=100)) * sec_per_yr
heat = physics.nuclearwasteheat.repo_2020_conservative.heat
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time / sec_per_yr, heat(time) / 1e3, lw=2, label="reference", color="k")

for r in range(n_refinements):
    dt = 30.0 / (2.0**r)
    time = np.linspace(0.0, t_end, int(t_end / dt) + 1) * sec_per_yr
    edges = np.append(0, time) / sec_per_yr
    ax.stairs(heat(time) / 1e3, edges, label=f"{dt=}", baseline=None, lw=1.5)
ax.set_xlabel("time / yrs")
ax.set_ylabel("heat / kW")
ax.legend()
fig.show()

# %% [markdown]
# The mesh and its boundaries are generated easily via gmsh and msh2vtu:

# %%
msh_path = temp_dir / "square.msh"
meshlib.rect_mesh(lengths=100.0, n_edge_cells=[10, 1], out_name=msh_path)
_ = msh2vtu.msh2vtu(msh_path, output_path=temp_dir, log_level="ERROR")

# %% [markdown]
# Let's run the different simulations with increasingly fine temporal
# discretization via ogs6py, extract the temperature evolution via vtuIO and
# plot it:

# %%
r_range = range(n_refinements)
results_tmax = [temp_dir / f"nuclear_decay_tmax_{r}.vtu" for r in r_range]
results_qmax = [temp_dir / f"nuclear_decay_qmax_{r}.vtu" for r in r_range]
fig, ax = plt.subplots(figsize=(8, 4))

for r in range(n_refinements):
    model = ogs.OGS(
        PROJECT_FILE=temp_dir / "default.prj",
        INPUT_FILE=studies.convergence.examples.nuclear_decay_prj,
    )
    dt = 30.0 / (2.0**r)
    model.replace_text(str(dt * sec_per_yr), ".//delta_t")
    model.write_input()
    script_path = Path(studies.convergence.examples.nuclear_decay_bc).parent
    ogs_args = f"-m {temp_dir} -o {temp_dir} -s {script_path}"
    model.run_model(write_logs=False, args=ogs_args)

    pvd_path = str(temp_dir / "nuclear_decay.pvd")
    pvd = meshlib.MeshSeries(pvd_path)
    result_qmax = pvd.read_closest(30 * sec_per_yr)
    result_tmax = pvd.read_closest(150 * sec_per_yr)
    result_qmax.add_field_data(dt, "timestep_size")
    result_tmax.add_field_data(dt, "timestep_size")
    result_qmax.save(results_qmax[r])
    result_tmax.save(results_tmax[r])

    pvdio = vtuIO.PVDIO(pvd_path, dim=2, interpolation_backend="vtk")
    max_temperature = propertylib.presets.temperature(
        pvdio.read_time_series("temperature", {"pt0": [0, 0, 0]})["pt0"]
    )
    ts = pvdio.timesteps / sec_per_yr
    ax.plot(ts, max_temperature, lw=1.5, label=f"{dt=}")
ax.set_xlabel("time / yrs")
ax.set_ylabel("max T / °C")
ax.legend()
fig.show()

# %% [markdown]
# Temperature convergence at maximum heat production (t=30 yrs)
# -------------------------------------------------------------
#
# The grid convergence at this timepoint deviates significantly from 1,
# meaning the convergence is suboptimal (at least on the left boundary where the
# heating happens). The chosen timesteps are still to coarse to reach an
# asymptotic rate of convergence. The model behavior at this early part of the
# simulation is still very dynamic and needs finer timesteps to be captured with
# great accuracy. Nevertheless, the maximum temperature converges (sublinearly)
# from overestimated values, as expected.

# %%
report_name = str(temp_dir / "report.ipynb")
studies.convergence.run_convergence_study(
    output_name=report_name,
    mesh_paths=results_qmax,
    topology_path=results_qmax[-3],
    property_name="temperature",
    refinement_ratio=2.0,
)
HTML(workflow.jupyter_to_html(report_name, show_input=False))

# %% [markdown]
# Temperature convergence at maximum temperature (t=150 yrs)
# ----------------------------------------------------------
#
# The temperature convergence at this timepoint is much closer to 1, signifying
# a good convergence behaviour. The temperature gradient at this point in time
# is already settled and can be the solution convergences to a good degree with
# the chosen timesteps. Despite the good grid convergence, the maximum
# temperature converges only sublinearly, but as expected, from underestimated
# values.

# %%
studies.convergence.run_convergence_study(
    output_name=report_name,
    mesh_paths=results_tmax,
    topology_path=results_tmax[-3],
    property_name="temperature",
    refinement_ratio=2.0,
)
HTML(workflow.jupyter_to_html(report_name, show_input=False))

# %%

# sphinx_gallery_start_ignore

# Removing the created files to keep the code repository clean for developers.
# If you want to use the created jupyter notebook further, skip this step.
rmtree(temp_dir)

# sphinx_gallery_end_ignore
