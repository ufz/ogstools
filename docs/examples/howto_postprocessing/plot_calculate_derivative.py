"""
Calculate derivatives
==============================
This example shows how to calculate and plot derivatives.
"""

# %%
import ogstools as ot
from ogstools import examples

# %% [markdown]
# First we load the example data

# %%
ms = examples.load_meshseries_THM_2D_PVD()

# %% [markdown]
# Next we *can* add the desired derivative to the MeshSeries, but this step is
# not mandatory if you only want to plot it.

# %%
ms.add_new_data(ot.variables.temperature.gradient)
data = ms.point_data[ot.variables.temperature.gradient]
print("timesteps: {}, points: {}, dimensions: {}".format(*data.shape))

# %% [markdown]
# For works just as with any other variable. Here we plot the gradient at the
# last timestep (-1).

# %%
fig = ot.plot.contourf(ms[-1], ot.variables.temperature.gradient)

# %% [markdown]
# Besides the `gradient` we can also compute the `divergence`. For verification
# we can compare the strain trace and the displacement divergence, which should
# be roughly the same:

# %%
fig = ot.plot.contourf(ms[-1], ot.variables.strain.trace)
u_div = ot.variables.displacement.divergence.replace(
    output_unit="%", bilinear_cmap=False
)
fig = ot.plot.contourf(ms[-1], u_div, cmap="coolwarm")

# %% [markdown]
# Finally, we can also analyze the `vorticity`. This will result in a vector,
# perpendicular to the input data. I.e. If we want to to inspect the `vorticity`
# of the fluid flow in the xy-plane, we have to look at its z-component:

# %%
fig = ot.plot.contourf(ms[-1], ot.variables.velocity.vorticity["z"])
