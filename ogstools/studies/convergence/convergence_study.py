# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Convergence study
#
# This script performs a convergence study and generates plots to analyze the
# convergence of numerical simulations.
# Below are the custom parameters for this report.

# %% tags=["parameters"]
mesh_paths: "pv.DataSet" = None
property_name: str = ""

# %% [markdown]
# Import required modules and setup plots.

# %%
import pyvista as pv  # noqa: E402

import ogstools.meshplotlib as mpl  # noqa: E402
from ogstools.propertylib import THM  # noqa: E402
from ogstools.studies.convergence import (  # noqa: E402
    convergence_metrics,
    plot_convergence,
    plot_convergence_errors,
    richardson_extrapolation,
)

mpl.setup.reset()
mpl.setup.show_element_edges = True
mpl.setup.ax_aspect_ratio = 1

if mesh_paths is None:
    exit()

# %% [markdown]
# Read the meshes, get Property object from property name, define topology and calculate Richardson extrapolation.

# %%
meshes = [pv.read(mesh_path) for mesh_path in mesh_paths]
mesh_property = THM.find_property(property_name)
mesh_property.output_unit = ""
mesh_property.data_unit = ""
topology = meshes[-3]
richardson = richardson_extrapolation(meshes, mesh_property, topology)

# %% [markdown]
# Plotting the grid convergence.

# %%
fig = mpl.plot(richardson, "grid_convergence")

# %% [markdown]
# Plotting the 3 finest discretizations.

# %%
fig = mpl.plot(meshes[-3:], mesh_property)

# %% [markdown]
# Plotting the Richardson extrapolation.

# %%
fig = mpl.plot(richardson, mesh_property)

# %% [markdown]
# Table of convergence metrics.

# %%
mpl.core.plt.rcdefaults()
metrics = convergence_metrics(meshes, richardson, mesh_property)
metrics.style.format("{:,.4g}").hide()

# %% [markdown]
# Absolute convergence metrics.

# %%
fig = plot_convergence(metrics, mesh_property)

# %% [markdown]
# Relative errors in loglog-scale.

# %%
fig = plot_convergence_errors(metrics)
