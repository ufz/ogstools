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
property_name: str = ""
refinement_ratio: float = 2.0
reference_solution_path = None

# %% tags=["remove_input"]
# Import required modules and customize plot settings.

import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

from ogstools import meshlib, meshplotlib, propertylib, studies  # noqa: E402

meshplotlib.setup.reset()
meshplotlib.setup.show_element_edges = True
meshplotlib.setup.fig_scale = 0.5
meshplotlib.setup.combined_colorbar = False

# %% tags=["remove_input"]
# Here, the meshes are read, a Property object is created from the property
# name and the Richardson extrapolation calculated.
# The 3 finest meshes of those provided will be used for the Richardson
# extrapolation.

mesh_series = [meshlib.MeshSeries(mesh_path) for mesh_path in mesh_paths]
timestep_sizes = [np.mean(np.diff(ms.timevalues)) for ms in mesh_series]
meshes = [ms.read_closest(timevalue) for ms in mesh_series]
topology: pv.UnstructuredGrid = meshes[-3]
mesh_property = propertylib.properties.get_preset(property_name, meshes[0])
richardson = studies.convergence.richardson_extrapolation(
    meshes, mesh_property, topology, refinement_ratio
)

# %% [markdown]
# ## Grid convergence
#
# If the shown values are approximately 1, this means that the results are in
# asymptotic range of convergence.

# %% tags=["remove_input"]
fig = meshplotlib.plot(richardson, "grid_convergence")

# %% [markdown]
# ## Grid comparison
#
# Visualizing the requested mesh property on the 3 finest discretizations:

# %% tags=["remove_input"]
fig = meshplotlib.plot(meshes[-3:], mesh_property)

# %% [markdown]
# ## Richardson extrapolation.
#
# Visualizing the Richardson extrapolation of the requested mesh property.
# If a reference solution is provided, the difference between the two is shown
# as well. Otherwise the difference between the finest discretization and
# the Richardson extrapolation is shown.

# %% tags=["remove_input"]
fig = meshplotlib.plot(richardson, mesh_property)

data_key = mesh_property.data_name
if reference_solution_path is None:
    diff_mesh = meshlib.difference(
        richardson, topology.sample(meshes[-1]), mesh_property
    )
    fig = meshplotlib.plot(diff_mesh, mesh_property)
else:
    reference_solution = topology.sample(
        meshlib.MeshSeries(reference_solution_path).read_closest(timevalue)
    )
    diff_mesh = meshlib.difference(
        reference_solution, richardson, mesh_property
    )
    fig = meshplotlib.plot(diff_mesh, mesh_property)

# %% [markdown]
# ## Convergence metrics

# %% tags=["remove_input"]
metrics = studies.convergence.convergence_metrics(
    meshes, richardson, mesh_property, timestep_sizes
)
metrics.style.format("{:,.5g}").hide()

# %% [markdown]
# ## Relative errors

# %% tags=["remove_input"]
meshplotlib.core.plt.rcdefaults()
fig = studies.convergence.plot_convergence_errors(metrics)

# %% [markdown]
# ## Absolute values

# %% tags=["remove_input"]
fig = studies.convergence.plot_convergence(metrics, mesh_property)
