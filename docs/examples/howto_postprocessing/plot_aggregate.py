"""
Aggregation of Meshseries Data
==============================

In this example we show how to aggregate data in a model over all timesteps as
well as plot differences between two timesteps. For this purpose we use
a component transport example from the ogs benchmark gallery
(https://www.opengeosys.org/stable/docs/benchmarks/hydro-component/elder_jupyter/).

To see this benchmark results over all timesteps have a look at
:ref:`sphx_glr_auto_examples_howto_plot_plot_animation.py`.
"""

# %%
import numpy as np

import ogstools as ot
from ogstools import examples

mesh_series = examples.load_meshseries_CT_2D_XDMF().scale(time="a")
saturation = ot.variables.saturation

# %% [markdown]
# To read your own data as a mesh series you can do:
#
# ..  code-block:: python
#
#   from ogstools.meshlib import MeshSeries
#   mesh_series = MeshSeries("filepath/filename_pvd_or_xdmf")
#
# You can also use a variable from the available presets instead of needing to
# create your own:
# :ref:`sphx_glr_auto_examples_howto_postprocessing_plot_variables.py`

# %% [markdown]
# Plotting
# --------
# You aggregate the data in MeshSeries over all timesteps given some
# aggregation function. The following are available:
# ``min``, ``max``, ``mean``, ``median``, ``sum``, ``std``, ``var``,
# ``argmin``, ``argmax``, ``time_of_min``, ``time_of_max``.
# The following code gets the maximum saturation for each point in the mesh over
# all timesteps and plots it.
#
# Due to numerical oscillations, this dataset contains a few cells with
# slightly unphysical saturation (below 0% or above 100%). We clip the
# colorbar at ``vmin=0, vmax=100`` to avoid these values dominating the
# plotted range.

# %%
fig = ot.plot.contourf(mesh_series, saturation.max, vmin=0, vmax=100)
fig.show()

# %% [markdown]
# For any function which is not covered you can create a new variable for
# further usage.

# %%
saturation.aggregate(np.nanmax)

# %% [markdown]
# If you want to have the data instead of a plot for further processing, you can
# create a new mesh, with the aggregated data in it, either by passing a
# variable and a numpy function or by passing a variable's aggregation directly:

# %%
mesh = mesh_series.aggregate_temporal(saturation, np.max)
# The following works the same way:
# mesh = mesh_series.aggregate_temporal(saturation.max)
print(list(mesh.point_data.keys()))

# %% [markdown]
# It is also possible to plot the time when the minimum or maximum occurs.

# %%
fig = ot.plot.contourf(mesh_series, saturation.time_of_max)
fig.show()

# %% [markdown]
# Likewise we can calculate and visualize the variance of the saturation:


# %%
mesh = mesh_series.aggregate_temporal(saturation.var)
fig = ot.plot.contourf(mesh, saturation.var)
fig.show()

# %% [markdown]
# Difference between the last and the first timestep:

# %%
mesh = ot.mesh.difference(mesh_series[-1], mesh_series[0], saturation)
fig = ot.plot.contourf(mesh, saturation.difference)
fig.show()

# %% [markdown]
# It's also possible to aggregate the data per timestep to return a timeseries
# of e.g. the max or mean value of a variable in the entire domain.

# %%
mean_saturation_array = mesh_series.aggregate_spatial(saturation.mean)

# %% [markdown]
# Instead of calculating the array itself, you can also plot it directly:

# %%
fig = mesh_series.plot_line(saturation.std)
fig.tight_layout()
fig.show()

# %% [markdown]
# Data inspection
# ---------------
# We can also use these aggregating variables to inspect the data directly.
# By default the aggregation occurs over the last axis (for vector and matrix
# data, the second last axis is chosen). Thus, the following code would first
# compute the variance of the data field per timestep and then find the maximum
# timestep index:

# %%
var_max_idx = saturation.var.argmax.transform(mesh_series)
print(f"timestep index of maximum variance {var_max_idx}")
var_max_time = saturation.var.time_of_max.transform(mesh_series)
print(f"timevalue of maximum variance {var_max_time}")
assert var_max_time == mesh_series.timevalues[var_max_idx]

# %% [markdown]
# And if you want to determine the point (or cell) index where
# the total maximum or minimum is located you can manipulate the axis of the
# aggregating function:

# %%
temporal_var = saturation.var.change_axis(0)
var_max_pt_idx = temporal_var.argmax.transform(mesh_series)
print(f"point index of maximum variance {var_max_pt_idx}")
var_max = temporal_var.max.transform(mesh_series)
print(f"maximum variance {var_max}")
assert var_max == mesh_series.values(temporal_var)[var_max_pt_idx]


# %%
