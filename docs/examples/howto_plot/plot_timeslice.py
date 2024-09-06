"""
How to Create Time Slices
=========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how to create a filled contourplot of transient data
over a sampling line. For this purpose we use a component transport example
from the ogs benchmark gallery
(https://www.opengeosys.org/docs/benchmarks/hydro-component/elder/).

To see this benchmark results over all timesteps have a look at
:ref:`sphx_glr_auto_examples_howto_plot_plot_animation.py`.
"""

# %% [markdown]
# Let's load the data and create 3 different lines to sample over:
# vertical, horizontal and diagonal.
import numpy as np

import ogstools as ogs
from ogstools import examples

mesh_series = examples.load_meshseries_CT_2D_XDMF()
si = ogs.variables.saturation
points_vert = np.linspace([25, 0, -75], [25, 0, 75], num=100)
points_hori = np.linspace([0, 0, 60], [150, 0, 60], num=100)
points_diag = np.linspace([25, 0, 75], [100, 0, 0], num=100)
fig = mesh_series.mesh(-1).plot_contourf(si, vmin=0)
fig.axes[0].plot(points_vert[[0, -1], 0], points_vert[[0, -1], 2], "-k2")
fig.axes[0].plot(points_hori[[0, -1], 0], points_hori[[0, -1], 2], "--k2")
fig.axes[0].plot(points_diag[[0, -1], 0], points_diag[[0, -1], 2], "-.k2")

# %% [markdown]
# The function plot_time_slice automatically detects if the line lies on a
# cardinal direction and labels the y-axes with the changing spatial dimension.
fig = mesh_series.plot_time_slice(si, points_vert, time_unit="a")

# %% [markdown]
# By default the plot is smoothened with interpolation. When deactivated, we
# see the raw sampled data. Be sure to adjust the number of sampling points if
# the MeshSeries contains a lot of timesteps.
fig = mesh_series.plot_time_slice(
    si, points_vert, time_unit="a", interpolate=False
)

# %% [markdown]
# The horizontal sampling line gets also labeled appropriately.
fig = mesh_series.plot_time_slice(si, points_hori, time_unit="a")

# %% [markdown]
# If the line doesn't point in a cardinal direction the distance along the
# line is used for the y-axis by default. You can however, specify if you want
# to use spatial dimension via the argument "y_axis". This may be useful when
# plotting data of an edge / boundary of the mesh.
fig = mesh_series.plot_time_slice(si, points_diag, time_unit="a")
