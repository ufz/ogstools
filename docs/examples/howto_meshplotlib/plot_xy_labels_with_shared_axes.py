"""
Labeling directional shared axes
=================================

.. sectionauthor:: Feliks Kiszkurno (Helmholtz Centre for Environmental Research GmbH - UFZ)

.. warning:: This example discusses functionality that may experience breaking changes in the near future!

For this example we load a 2D meshseries dataset from within the ``meshplotlib`` examples.
This tutorial covers automatic labeling the directional axes (X and Y) under various conditions (shared and nor shared X and Y axes).
"""

# %%
# Import Python packages, change some settings and load example data set
import matplotlib.pyplot as plt
import numpy as np

import ogstools.meshplotlib as mpl
from ogstools.meshplotlib import (
    clear_labels,
    examples,
    label_spatial_axes,
    plot,
    plot_diff,
)
from ogstools.propertylib import presets

plt.rcParams.update({"font.size": 32})

mpl.setup.reset()
mpl.setup.length.output_unit = "km"

meshseries = examples.meshseries_THM_2D

# %%
# First, by default (without shared axes) both X and Y axes will be labeled automatically. The default is that both axes are shared and this will be respected.

mpl.setup.combined_colorbar = False
fig = plot([meshseries.read(0), meshseries.read(1)], presets.temperature)


# %%
# On user provided figure and axis, this behaviour is different. To allow for more complex combinations of plot functions, meshseries and process variables, the axis belonging to specific subplot has to be passed. In this case the default is to plot labels on each axis regardless of whether it is share or not.

mpl.setup.combined_colorbar = False
fig, ax = plt.subplots(2, 2, figsize=(40, 20), sharex=True, sharey=True)
plot(meshseries.read(0), presets.temperature, fig=fig, ax=ax[0][0])
plot(meshseries.read(1), presets.temperature, fig=fig, ax=ax[1][0])
plot_diff(
    meshseries.read(0),
    meshseries.read(1),
    presets.temperature,
    fig=fig,
    ax=ax[0][1],
)
plot_diff(
    meshseries.read(1),
    meshseries.read(0),
    presets.temperature,
    fig=fig,
    ax=ax[1][1],
)
fig.tight_layout()

# %%
# If user wishes to have labels respecting shared axes, they need to be first removed and applied again. Meshplotlib provides two function that make it easy: clear_labels and label_spatial_axes. They have to be called after the last plot related function call.

mpl.setup.combined_colorbar = False
fig, ax = plt.subplots(2, 2, figsize=(40, 20), sharex=True, sharey=True)
plot(meshseries.read(0), presets.temperature, fig=fig, ax=ax[0][0])
plot(meshseries.read(1), presets.temperature, fig=fig, ax=ax[1][0])
plot_diff(
    meshseries.read(0),
    meshseries.read(1),
    presets.temperature,
    fig=fig,
    ax=ax[0][1],
)
plot_diff(
    meshseries.read(1),
    meshseries.read(0),
    presets.temperature,
    fig=fig,
    ax=ax[1][1],
)
ax = clear_labels(ax)
ax = label_spatial_axes(ax, np.array([0, 1]))
fig.tight_layout()
