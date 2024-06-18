"""
Visualizing 2D model data
=========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

To demonstrate the creation of filled contour plots we load a 2D THM meshseries
example. In the ``plot.setup`` we can provide a dictionary to map names
to material ids. First, let's plot the material ids (cell_data). Per default in
the setup, this will automatically show the element edges.
"""

# %%
import ogstools as ot
from ogstools import examples

ot.plot.setup.material_names = {i + 1: f"Layer {i+1}" for i in range(26)}
mesh = examples.load_meshseries_THM_2D_PVD().read(1)

# %% [markdown]
# To read your own data as a mesh series you can do:
#
# ..  code-block:: python
#
#   mesh_series = ot.MeshSeries("filepath/filename_pvd_or_xdmf")
#

# %%
fig = mesh.plot_contourf(ot.properties.material_id)

# %% [markdown]
# Now, let's plot the temperature field (point_data) at the first timestep.
# The default temperature property from the `propertylib` reads the temperature
# data as Kelvin and converts them to degrees Celsius.

# %%
fig = mesh.plot_contourf(ot.properties.temperature)

# %% [markdown]
# We can also plot components of vector properties:

# %%
fig = mesh.plot_contourf(ot.properties.displacement[0])

# %%
fig = mesh.plot_contourf(ot.properties.displacement[1])

# %% [markdown]
# This example has hydraulically deactivated subdomains:

# %%
fig = mesh.plot_contourf(ot.properties.pressure.get_mask())

# %% [markdown]
# Let's plot the fluid velocity field.

# %%
fig = mesh.plot_contourf(ot.properties.velocity)

# %% [markdown]
# Let's plot it again, this time log-scaled.

# %%
ot.plot.setup.log_scaled = True
ot.plot.setup.p_min = -8
fig = mesh.plot_contourf(ot.properties.velocity)
