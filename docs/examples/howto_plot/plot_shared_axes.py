"""
Shared axes
===========

.. sectionauthor:: Feliks Kiszkurno (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how to create filled contourplots on a figure with
subplots having shared axes.
"""

# %%
import matplotlib.pyplot as plt

import ogstools as ot
from ogstools import examples

meshseries = examples.load_meshseries_THM_2D_PVD().scale(spatial=("m", "km"))
mesh_0 = meshseries.mesh(0)
mesh_1 = meshseries.mesh(1)
temperature = ot.variables.temperature

# %%
# If you pass multiple meshes to :py:func:`ogstools.plot.contourf`
# by default both x and y axes will shared. Thus, only the outer axes get
# axes labels and tick label.

fig = ot.plot.contourf([mesh_0, mesh_1], temperature)

# %%
# On user defined figure and axis the axis belonging to specific subplot has to
# be passed.

fig, axs = plt.subplots(2, 2, figsize=(40, 17), sharex=True, sharey=True)
diff_a = mesh_0.difference(mesh_1, temperature)
diff_b = mesh_1.difference(mesh_0, temperature)
ot.plot.contourf(mesh_0, temperature, fig=fig, ax=axs[0][0])
ot.plot.contourf(mesh_1, temperature, fig=fig, ax=axs[1][0])
ot.plot.contourf(diff_a, temperature, fig=fig, ax=axs[0][1])
ot.plot.contourf(diff_b, temperature, fig=fig, ax=axs[1][1])
fig.tight_layout()
