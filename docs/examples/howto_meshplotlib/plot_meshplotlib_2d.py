"""
Visualizing 2D model data
=========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

For this example we load a 2D meshseries dataset from within the ``meshplotlib`` examples.
In the ``meshplotlib.setup`` we can provide a dictionary to map names to material ids.
First, let's plot the material ids (cell_data). Per default in the setup, this
will automatically show the element edges.
"""

# %%
import ogstools.meshplotlib as mpl
from ogstools.meshplotlib.examples import meshseries_THM_2D
from ogstools.propertylib import THM

mpl.setup.reset()
mpl.setup.length.output_unit = "km"
mpl.setup.material_names = {i + 1: f"Layer {i+1}" for i in range(26)}
mesh = meshseries_THM_2D.read(1)
fig = mpl.plot(mesh, property=THM.material_id)

# %%
# Now, let's plot the temperature field (point_data) at the first timestep.
# The default temperature property from the `propertylib` reads the temperature
# data as Kelvin and converts them to degrees Celsius.

fig = mpl.plot(mesh, property=THM.temperature)

# %%
# This example has hydraulically deactivated subdomains:

fig = mpl.plot(mesh, THM.pressure.get_mask())

# %%
# Let's plot the fluid velocity field on the hydraulically active part of the model.
mesh_H = mesh.threshold((1, 3), "MaterialIDs")
fig = mpl.plot(mesh_H, THM.velocity)

# %%
# It is also possible to plot a shape on top, e.g. to display an overburden.
mpl.plot_on_top(fig.axes[0], mesh_H, lambda x: min(max(0, 0.1 * (x - 3)), 100))
fig  # noqa: B018


# %%
# We can also plot components of vector property:

fig = mpl.plot(mesh, THM.displacement[0])

# %%
fig = mpl.plot(mesh, THM.displacement[1])
