"""
How to plot data at observation points
======================================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we plot the data values on observation points over all
timesteps. Since requested observation points don't necessarily coincide with
actual nodes of the mesh different interpolation options are available. See
:py:mod:`ogstools.meshlib.mesh_series.MeshSeries.probe` for more details.
Here again, we use a component transport example from the ogs benchmark gallery
(https://www.opengeosys.org/docs/benchmarks/hydro-component/elder/).
"""

# %%

# sphinx_gallery_start_ignore

# sphinx_gallery_thumbnail_number = 2

# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import numpy as np

from ogstools import meshplotlib
from ogstools.meshplotlib.examples import meshseries_CT_2D as mesh_series
from ogstools.propertylib import Scalar

meshplotlib.setup.reset()
si = Scalar(
    data_name="Si", data_unit="", output_unit="%", output_name="Saturation"
)
# alternatively:
# from ogstools.meshlib import MeshSeries
# mesh_series = MeshSeries("filepath/filename_pvd_or_xdmf")

# %% [markdown]
# Let's define 4 observation points and plot them on the mesh.

# %%
points = np.asarray([[x, 0, 60] for x in [0, 40, 80, 120]])
fig = meshplotlib.plot(mesh_series.read(0), si)
fig.axes[0].scatter(points[:, 0], points[:, 2], s=50, fc="none", ec="r", lw=3)
for i, point in enumerate(points):
    fig.axes[0].annotate(str(i), (point[0], point[2] - 5), va="top")
plt.rcdefaults()

# %% [markdown]
# And now probe the points and the values over time:

# %%
fig = meshplotlib.plot_probe(
    mesh_series=mesh_series, points=points, mesh_property=si, time_unit="a"
)

# %% [markdown]
# You can also pass create your own matplotlib figure and pass the axes object.
# Additionally, you can pass any keyword arguments which are known by
# matplotlibs plot function to further customize the curves.
# In this case `marker` and `linewidth` are not part of the API of `plot_probe`
# but get processed correctly anyway.

# %%
fig, axs = plt.subplots(nrows=2, figsize=[10, 5])
meshplotlib.plot_probe(
    mesh_series, points, si, time_unit="a", ax=axs[0], colors=["k", "b"],
    marker="."  # fmt: skip
)
points_2 = np.asarray([[50, 0, z] for z in [60, 50, 40, 30]])
meshplotlib.plot_probe(
    mesh_series, points_2, si, time_unit="a", ax=axs[1], linestyles=["-"],
    linewidth=1  # fmt: skip
)
