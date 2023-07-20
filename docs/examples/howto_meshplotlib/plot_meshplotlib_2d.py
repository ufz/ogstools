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
fig = mpl.plot(meshseries_THM_2D.read(0), property=THM.material_id)

# %%
# Now, let's plot the temperature field (point_data) at the first timestep.
# The default temperature property from the `propertylib` reads the temperature
# data as Kelvin and converts them to degrees Celsius.

fig = mpl.plot(meshseries_THM_2D.read(1), property=THM.temperature)

# %%
# This example has hydraulically deactivated subdomains:

fig = mpl.plot(meshseries_THM_2D.read(1), THM.pressure.get_mask())

# %%
# Let's plot the fluid velocity field on the hydraulically active part of the model.

fig = mpl.plot(
    meshseries_THM_2D.read(1).threshold((1, 3), "MaterialIDs"), THM.velocity
)

# %%
# We can also plot components of vector property:

fig = mpl.plot(meshseries_THM_2D.read(1), THM.displacement[0])

# %%
fig = mpl.plot(meshseries_THM_2D.read(1), THM.displacement[1])
