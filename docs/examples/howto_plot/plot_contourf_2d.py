"""
Visualizing 2D model data
=========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

To demonstrate the creation of filled contour plots we load a 2D THM meshseries
example. In the ``plot.setup`` we can provide a dictionary to map names
to material ids. Other plot configurations are also available, see:
:py:obj:`ogstools.plot.plot_setup.PlotSetup`. Some of these options are also
available as keyword arguments in the function call. Please see
:py:obj:`ogstools.plot.contourplots.contourf` for more information.
"""

# %%
import ogstools as ogs
from ogstools import examples

ogs.plot.setup.material_names = {i + 1: f"Layer {i+1}" for i in range(26)}
mesh = examples.load_meshseries_THM_2D_PVD().mesh(1)

# %% [markdown]
# To read your own data as a mesh series you can do:
#
# ..  code-block:: python
#
#   mesh_series = ogs.MeshSeries("filepath/filename_pvd_or_xdmf")
#

# %% First, let's plot the material ids (cell_data). Per default in
# the setup, this will automatically show the element edges.
fig = mesh.plot_contourf(ogs.variables.material_id)

# %% [markdown]
# Now, let's plot the temperature field (point_data) at the first timestep.
# The default temperature variable from the `variables` reads the temperature
# data as Kelvin and converts them to degrees Celsius.

# %%
fig = mesh.plot_contourf(ogs.variables.temperature, show_max=True)

# %% [markdown]
# We can also plot components of vector variables:

# %%
fig = mesh.plot_contourf(
    ogs.variables.displacement[0], show_min=True, show_max=True
)

# %%
fig = mesh.plot_contourf(
    ogs.variables.displacement[1], show_max=True, show_edges=True
)

# %% [markdown]
# This example has hydraulically deactivated subdomains:

# %%
fig = mesh.plot_contourf(ogs.variables.pressure.get_mask(), fontsize=40)

# %% [markdown]
# Let's plot the fluid velocity field.

# %%
fig = mesh.plot_contourf(ogs.variables.velocity, show_region_bounds=False)

# %% [markdown]
# Let's plot it again, this time log-scaled.

# %%
fig = mesh.plot_contourf(ogs.variables.velocity, log_scaled=True, vmin=-8)
