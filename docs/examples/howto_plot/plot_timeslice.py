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
# Let's load the data which we want to investigate.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import ogstools as ot
from ogstools import examples

mesh_series = examples.load_meshseries_CT_2D_XDMF().scale(time=("s", "a"))
y = mesh_series[0].center[1]  # flat y coordinate of this 2D mesh is not 0
si = ot.variables.saturation

# %% [markdown]
# Now we setup two sampling lines.

# %%
pts_vert = np.linspace([25, y, -75], [25, y, 75], num=100)
pts_diag = np.linspace([25, y, 75], [100, y, 0], num=100)
fig = mesh_series.mesh(-1).plot_contourf(si, vmin=0)
fig.axes[0].plot(pts_vert[:, 0], pts_vert[:, 2], "-k", linewidth=3)
fig.axes[0].plot(pts_diag[:, 0], pts_diag[:, 2], "-.k", linewidth=3)

# %% [markdown]
# Here, we first show a regular line sample plot for the vertical sampling line
# for each timestep.

# %%
fig, ax = plt.subplots(figsize=[15, 8])
for mesh, timevalue in zip(mesh_series, mesh_series.timevalues, strict=True):
    sample = pv.PolyData(pts_vert).sample(mesh)
    color = str(0.8 * timevalue / mesh_series.timevalues[-1])
    label = f"{timevalue:.1f} a"
    fig = ot.plot.line(
        sample, si, "z", ax=ax, label=label, color=color, fontsize=20
    )
# %% [markdown]
# As the above kind of plot is getting cluttered for lots of timesteps we
# provide a function to create a filled contour plot over the transient data.
# The function :meth:`~ogstools.meshlib.mesh_series.MeshSeries.plot_time_slice`
# automatically detects if the line lies on a cardinal direction and labels the
# y-axes with the changing spatial dimension.

# %%
fig = mesh_series.plot_time_slice(si, pts_vert)

# %% [markdown]
# By default the plot is smoothened with interpolation. When deactivated, we
# see the edges of the raw sampled data. When using the interpolation, be sure
# to adjust the number of sampling points if the MeshSeries contains a lot of
# small timesteps.
fig = mesh_series.plot_time_slice(si, pts_vert, interpolate=False)

# %% [markdown]
# If the line doesn't point in a cardinal direction the distance along the
# line is used for the y-axis by default. You can however, specify if you want
# to use spatial dimension via the argument "y_axis". This may be useful when
# plotting data of an edge / boundary of the mesh.
fig = mesh_series.plot_time_slice(si, pts_diag)
