"""
How to plot data at observation points
======================================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we plot the data values on observation points over all
timesteps. Since requested observation points don't necessarily coincide with
actual nodes of the mesh different interpolation options are available. See
:py:mod:`ogstools.meshlib.mesh_series.MeshSeries.probe` for more details.
Here we use a component transport example from the ogs benchmark gallery
(https://www.opengeosys.org/docs/benchmarks/hydro-component/elder/).
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = 2
# fmt:off

# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import numpy as np

import ogstools as ogs
from ogstools import examples

mesh_series = examples.load_meshseries_CT_2D_XDMF()
si = ogs.variables.saturation

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
# Let's define 4 observation points and plot them on the mesh.

# %%
points = np.asarray(
    [[x, 0, 60] for x in [0, 40, 80, 120]]
    + [[x, 0, 40] for x in [0, 40, 80, 120]]
)
fig = mesh_series.mesh(0).plot_contourf(si)
fig.axes[0].scatter(points[:, 0], points[:, 2], s=50, fc="none", ec="r", lw=3)
for i, point in enumerate(points):
    fig.axes[0].annotate(str(i), (point[0], point[2] - 5), va="top")
plt.rcdefaults()  # TODO

# %% [markdown]
# And now probe the points and the values over time:

# %%,
labels = [f"{i}: {label}" for i, label in enumerate(ogs.plot.utils.justified_labels(points))]
fig = mesh_series.plot_probe(
    points=points[:4], variable=si, time_unit="a", labels=labels[:4]
)
# %% [markdown]
# You can also pass create your own matplotlib figure and pass the axes object.
# Additionally, you can pass any keyword arguments which are known by
# matplotlibs plot function to further customize the curves.
# In this case `marker` and `linewidth` are not part of the API of `plot_probe`
# but get processed correctly anyway.
# If you want to have more freedom with the data you can just do the probing,
# adapt to your needs and then do the plotting yourself:

# %%
fig, axs = plt.subplots(nrows=2, figsize=[10, 5])
mesh_series.plot_probe(
    points[:4], si, time_unit="a", ax=axs[0], colors=["k"],
    labels=labels[:4], marker=".")
mesh_series.plot_probe(
    points[4:], si, time_unit="a", ax=axs[1], linestyles=["-"],
    labels=labels[4:], linewidth=1,
)
# add the mean of the observation point timeseries
values = si.transform(mesh_series.probe(points, data_name=si.data_name))
mean_values = np.mean(values.reshape((-1, 2, 4)), axis=-1)
ts = mesh_series.timevalues("a")
for index in range(2):
    fig.axes[index].plot(ts, mean_values[:, index], "rk"[index], label="mean")
    fig.axes[index].legend()
