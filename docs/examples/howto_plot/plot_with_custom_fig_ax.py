"""
Plotting different process variables on already existing Matplotlib figures / axes
==================================================================================

.. sectionauthor:: Feliks Kiszkurno (Helmholtz Centre for Environmental Research GmbH - UFZ)

This tutorial covers plotting meshseries data using user defined matplotlib
objects for figure and / or axes. This is useful if different plotting functions
ogstools are to be used on different subplots within the same figure.
"""

# %%
import matplotlib.pyplot as plt

import ogstools as ot
from ogstools import examples

meshseries = examples.load_meshseries_THM_2D_PVD().scale(spatial=("m", "km"))

ot.plot.setup.combined_colorbar = False

# %% [markdown]
# Compare different variables
# ----------------------------

# %%
# It is possible to plot various variables in different subplots of the
# same figure:

fig, ax = plt.subplots(2, 1, figsize=(15, 15))
meshseries.mesh(0).plot_contourf(ot.variables.temperature, fig=fig, ax=ax[0])
meshseries.mesh(1).plot_contourf(ot.variables.displacement, fig=fig, ax=ax[1])
fig.tight_layout()

# %% [markdown]
# Plot two time steps and their difference
# ----------------------------------------

# %%
# We can use the same approach to plot the difference between different time
# steps can be plotted. Color bars can be drawn automatically, if user provides
# both Figure and Axes objects:

fig, ax = plt.subplots(3, 1, figsize=(20, 30))

meshseries.mesh(0).plot_contourf(ot.variables.temperature, fig=fig, ax=ax[0])
meshseries.mesh(1).plot_contourf(ot.variables.temperature, fig=fig, ax=ax[1])
diff_mesh = meshseries.mesh(1).difference(
    meshseries.mesh(0), ot.variables.temperature
)


diff_mesh.plot_contourf(ot.variables.temperature, fig=fig, ax=ax[2])
ax[0].set_title(r"$T(\mathrm{t}_{0})$")
ax[1].set_title(r"$T(\mathrm{t}_{end})$")
ax[2].set_title(r"$T(\mathrm{t}_{end})$-$T(\mathrm{t}_{0})$")
fig.tight_layout()
