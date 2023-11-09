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
topology_path = None
timestep: int = 0
property_name: str = ""
refinement_ratio: float = None
reference_solution_path = None

# %% tags=["remove_input"]
# Import required modules and customize plot settings.
from copy import deepcopy  # noqa: E402

import pyvista as pv  # noqa: E402

from ogstools import meshlib, meshplotlib, propertylib, studies  # noqa: E402

meshplotlib.setup.reset()
meshplotlib.setup.show_element_edges = True
meshplotlib.setup.fig_scale = 0.5
meshplotlib.setup.combined_colorbar = False

# %% tags=["remove_input"]
# Here, the meshes are read, a Property object is created from the property
# name, the topology is read and the Richardson extrapolation calculated.
# The 3 finest meshes of those provided will be used for the Richardson
# extrapolation.

meshes: list[pv.DataSet] = []
for mesh_path in mesh_paths:
    if mesh_path.split(".")[-1] in ["xdmf", "pvd"]:
        meshes += [meshlib.MeshSeries(mesh_path).read(timestep)]
    else:
        meshes += [pv.read(mesh_path)]
topology: pv.DataSet = pv.read(topology_path)
if property_name in meshes[0].point_data:
    data_shape = meshes[0][property_name].shape
else:
    data_shape = None
mesh_property = propertylib.presets.resolve_property(property_name, data_shape)
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
# Visualizing the Richardson extrapolation of the requested mesh property on
# the chosen topology. If a reference solution is provided, the difference
# between the two is shown as well. Otherwise the difference between the
# finest discretization and the Richardson extrapolation is shown.

# %% tags=["remove_input"]
fig = meshplotlib.plot(richardson, mesh_property)

if reference_solution_path is None:
    reference_solution = richardson
    diff = deepcopy(topology.sample(meshes[-1]))
else:
    if reference_solution_path.split(".")[-1] in ["xdmf", "pvd"]:
        reference_solution = meshlib.MeshSeries(reference_solution_path).read(
            timestep
        )
    else:
        reference_solution = pv.read(reference_solution_path)
    diff = deepcopy(richardson)
diff["difference"] = (
    reference_solution[mesh_property.data_name] - diff[mesh_property.data_name]
)
diff_unit = (mesh_property(1) - mesh_property(1)).units
diff_property = type(mesh_property)(
    data_name="difference",
    data_unit=diff_unit,
    output_unit=diff_unit,
    output_name=mesh_property.output_name + " difference",
)
fig = meshplotlib.plot(diff, diff_property)

# %% [markdown]
# ## Convergence metrics

# %% tags=["remove_input"]
metrics = studies.convergence.convergence_metrics(
    meshes, richardson, mesh_property
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
