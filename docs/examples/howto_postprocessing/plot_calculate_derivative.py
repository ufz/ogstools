"""
Calculate derivatives
==============================
This example shows how to calculate and plot derivatives over Mesh and
MeshSeries.
"""

# %%
import ogstools as ot
from ogstools import examples
from ogstools.mesh.derivatives import calculate_derivative

# %% [markdown]
# First we load the example data and define variables of interest

# %%
ms = examples.load_meshseries_THM_2D_PVD()
temperature = ot.variables.temperature
velocity = ot.variables.velocity

# %% [markdown]
# Next we can add the desired derivative to the MeshSeries (ms), here
# it is the temperature gradient:

# %%
ms.add_derivative(temperature.gradient)

# %% [markdown]
# Now gradient can be plotted for any time step or as in following example,
# the last one (-1). The derivative can be selected for plotting by string:

# %%
fig_grad_T = ot.plot.contourf(ms.mesh(-1), "temperature_gradient")

# %% [markdown]
# or by variable.output_name property

# %%
fig_grad_T = ot.plot.contourf(ms.mesh(-1), temperature.gradient.data_name)

# %% [markdown]
# The same can be done for a specific time step in MeshSeries:

# %%
calculate_derivative(ms[-2], velocity.divergence)

# %% [markdown]
# Note, that since velocity is a Vector, only its magnitude can be plotted.

# %%
fig_div_v_mesh = ot.plot.contourf(
    ms[-2],
    velocity.divergence.data_name,
    # "darcy_velocity_divergence" # Alternative
)

# %% [markdown]
# In the same way more complex variables can be used.
# For example following line will take stress as input, calculate von Mises
# seres and then its gradient:
# ```
# ms[-1].add_derivative(ot.variables.stress.von_Mises.gradient)
# ```
# But it is possible to avoid this intermediary step and plot the data directly
# (without storing it in Mesh or MeshSeries objects):

# %%
fig_von_Mises = ot.plot.contourf(
    ms[-1], ot.variables.stress.von_Mises.gradient.magnitude.data_name
)

# %%
