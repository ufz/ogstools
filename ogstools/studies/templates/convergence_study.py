# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---
# %% tags=["remove_cell"]
# mypy: ignore-errors
#
# This script is intended to be run via papermill with custom parameters.

# %% tags=["parameters", "remove_cell"]
# These are placeholders and get replaced with injected parameters.
mesh_paths = [""]
timevalue: float = 0.0
variable_name: str = ""
refinement_ratio: float = 2.0
reference_solution_path = None

# %% tags=["remove_input"]
# Import required modules and customize plot settings.
# pylint:disable=C0413
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

from ogstools import meshlib, plot, studies, variables  # noqa: E402

plot.setup.reset()
plot.setup.show_element_edges = True
plot.setup.combined_colorbar = False

# %% tags=["remove_input"]
# Here, the meshes are read, a Variable object is created from the variable
# name and the Richardson extrapolation calculated.
# The 3 finest meshes of those provided will be used for the Richardson
# extrapolation.

mesh_series = [meshlib.MeshSeries(mesh_path) for mesh_path in mesh_paths]
timestep_sizes = [np.mean(np.diff(ms.timevalues)) for ms in mesh_series]
meshes = [ms.mesh(ms.closest_timestep(timevalue)) for ms in mesh_series]
topology: pv.UnstructuredGrid = meshes[-3]
variable = variables.get_preset(variable_name, meshes[0])
richardson = studies.convergence.richardson_extrapolation(
    meshes, variable, topology, refinement_ratio
)

# %% [markdown]
# ## Grid convergence
#
# If the shown values are approximately 1, this means that the results are in
# asymptotic range of convergence.

# %% tags=["remove_input"]
fig = plot.contourf(richardson, "grid_convergence")

# %% [markdown]
# ## Grid comparison
#
# Visualizing the requested mesh variable on the 3 finest discretizations:

# %% tags=["remove_input"]
fig = plot.contourf(meshes[-3:], variable)

# %% [markdown]
# ## Richardson extrapolation
#
# Visualizing the Richardson extrapolation of the requested mesh variable.
# If a reference solution is provided, the difference between the two is shown
# as well. Otherwise the difference between the finest discretization and
# the Richardson extrapolation is shown.

# %% tags=["remove_input"]
fig = plot.contourf(richardson, variable)

data_key = variable.data_name
if reference_solution_path is None:
    diff_mesh = meshlib.difference(
        richardson, topology.sample(meshes[-1]), variable
    )
    fig = plot.contourf(diff_mesh, variable)
else:
    ms = meshlib.MeshSeries(reference_solution_path)
    timestep = ms.closest_timestep(timevalue)
    reference_solution = topology.sample(ms.mesh(timestep))
    diff_mesh = meshlib.difference(reference_solution, richardson, variable)
    fig = plot.contourf(diff_mesh, variable)

# %% [markdown]
# ## Convergence metrics

# %% tags=["remove_input"]
metrics = studies.convergence.convergence_metrics(
    meshes, richardson, variable, timestep_sizes
)
metrics.style.format("{:,.5g}").hide()

# %% [markdown]
# ## Relative errors

# %% tags=["remove_input"]
plot.contourplots.plt.rcdefaults()
fig = studies.convergence.plot_convergence_errors(metrics)

# %% [markdown]
# ## Absolute values

# %% tags=["remove_input"]
fig = studies.convergence.plot_convergence(metrics, variable)
