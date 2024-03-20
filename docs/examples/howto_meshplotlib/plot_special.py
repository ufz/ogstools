"""
How to plot limits and differences
==================================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we plot the limits of all nodes in a model through all
timesteps, as well as differences between to timesteps. For this purpose we use
a component transport example from the ogs benchmark gallery
(https://www.opengeosys.org/docs/benchmarks/hydro-component/elder/).

To see this benchmark results over all timesteps have a look at
:ref:`sphx_glr_auto_examples_howto_meshplotlib_plot_animation.py`.
"""

# %%
from ogstools.meshlib import difference
from ogstools.meshplotlib import examples, plot, plot_limit, setup
from ogstools.propertylib import Scalar

setup.reset()
mesh_series = examples.meshseries_CT_2D
si = Scalar(
    data_name="Si", data_unit="", output_unit="%", output_name="Saturation"
)
# alternatively:
# from ogstools.meshlib import MeshSeries
# mesh_series = MeshSeries("filepath/filename_pvd_or_xdmf")

# %% [markdown]
# Maximum though all timesteps

# %%
fig = plot_limit(mesh_series, si, "max")

# %% [markdown]
# Minimum though all timesteps

# %%
fig = plot_limit(mesh_series, si, "min")


# %% [markdown]
# Difference between the last and he first timestep:

# %%
diff_mesh = difference(si, mesh_series.read(-1), mesh_series.read(0))
fig = plot(diff_mesh, si.delta)
