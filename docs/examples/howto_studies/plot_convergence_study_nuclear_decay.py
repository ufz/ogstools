"""
Convergence study (temporal refinement)
=======================================

This example shows one possible implementation of how to do a convergence study
with temporal refinement. For this, a simple model using a time dependent heat
source on one side and constant temperature on the opposite side was set up.
The heat source is generated with the
:py:mod:`ogstools.physics.nuclearwasteheat` model.

Here is some theoretical background for the topic of grid convergence:

`Nasa convergence reference
<https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html>`_

`More comprehensive reference
<https://curiosityfluids.com/2016/09/09/establishing-grid-convergence/>`_

At least three meshes from simulations of increasing temporal refinement are
required for the convergence study. The topology has to stay the same.

The results to analyze are generated on the fly with the following code. If you
are only interested in the convergence study, please skip to
`Temperature convergence at maximum heat production (t=30 yrs)`_.

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
from scipy.constants import Julian_year as sec_per_yr

from ogstools import meshlib, msh2vtu, physics, propertylib, studies, workflow

temp_dir = Path(mkdtemp(prefix="nuclear_decay"))

# %% [markdown]
# Let's Visualize the temporal evolution of the source term and it's
# discretization in the simulations. We see, that with coarse time steps, the
# applied heat will be overestimated at first and underestimated once the heat
# curve has reached its maximum. The same is true for the resulting temperature.

# %%
n_refinements = 4
time_step_sizes = [30.0 / (2.0**r) for r in range(n_refinements)]
t_end = 180.0
time = np.append(0.0, np.geomspace(1.0, t_end, num=100)) * sec_per_yr
heat = physics.nuclearwasteheat.repo_2020_conservative.heat
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time / sec_per_yr, heat(time) / 1e3, lw=2, label="reference", color="k")

for dt in time_step_sizes:
    time = np.linspace(0.0, t_end, int(t_end / dt) + 1) * sec_per_yr
    edges = np.append(0, time) / sec_per_yr
    ax.stairs(heat(time) / 1e3, edges, label=f"{dt=}", baseline=None, lw=1.5)
ax.set_xlabel("time / yrs")
ax.set_ylabel("heat / kW")
ax.legend()
fig.show()

# %% [markdown]
# The mesh and its boundaries are generated easily via gmsh and
# :py:mod:`ogstools.msh2vtu`:

# %%
msh_path = temp_dir / "square.msh"
meshlib.rect(lengths=100.0, n_edge_cells=[10, 1], out_name=msh_path)
_ = msh2vtu.msh2vtu(msh_path, output_path=temp_dir, log_level="ERROR")

# %% [markdown]
# Let's run the different simulations with increasingly fine temporal
# discretization via ogs6py, extract the temperature evolution via vtuIO and
# plot it:

# %% tags=[toggle]
fig, ax = plt.subplots(figsize=(8, 4))
pvds = []

for dt in time_step_sizes:
    model = ogs.OGS(
        PROJECT_FILE=temp_dir / "default.prj",
        INPUT_FILE=studies.convergence.examples.nuclear_decay_prj,
    )
    model.replace_text(str(dt * sec_per_yr), ".//delta_t")
    prefix = "nuclear_decay_" + str(dt * sec_per_yr)
    model.replace_text(prefix, ".//prefix")
    model.write_input()
    script_path = Path(studies.convergence.examples.nuclear_decay_bc).parent
    ogs_args = f"-m {temp_dir} -o {temp_dir} -s {script_path}"
    model.run_model(write_logs=False, args=ogs_args)

    pvd_path = str(temp_dir / (prefix + ".pvd"))
    pvds += [pvd_path]
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
    mesh_paths=pvds,
    timevalue=30 * sec_per_yr,
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
    mesh_paths=pvds,
    timevalue=150 * sec_per_yr,
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
