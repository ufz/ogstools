"""
How to create Animations
========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

To demonstrate the creation of an animated plot we use a component transport
example from the ogs benchmark gallery
(https://www.opengeosys.org/docs/benchmarks/hydro-component/elder/).
"""

# %%
import numpy as np

import ogstools as ot
from ogstools import examples

ms = examples.load_meshseries_CT_2D_XDMF().scale(time=("s", "yrs"))
saturation = ot.variables.saturation

# %% [markdown]
# To read your own data as a mesh series you can do:
#
# ..  code-block:: python
#
#   from ogstools.meshlib import MeshSeries
#   mesh_series = MeshSeries("filepath/filename_pvd_or_xdmf")

# %% [markdown]
# You can choose which timesteps to render by either slicing the MeshSeries or
# by resampling it to new timesteps (this interpolates between existing
# timesteps).

# %%
# this would only use every second timesteps for animation
# ms = ms[::2]
# this uses equally spaced timesteps for a smooth animation
timevalues = np.linspace(ms.timevalues[0], ms.timevalues[-1], num=21)
ms = ot.MeshSeries.resample(ms, timevalues)

# %% [markdown]
# Now, let's animate the saturation solution. Note that rendering many frames in
# conjunction with large meshes might take a really long time. You need to setup
# a matplotlib figure first and a function, which is executed on each frame.
# This function has to take the individual values of the sequences passed as
# additional arguments, in this case the timevalues and the MeshSeries.


# %%
# clip to the right half
ms_r = ms.transform(lambda mesh: mesh.clip("-x", [-1, 0, 0]))

# create initial figure with fixed colorbar
fig = ot.plot.contourf(ms_r[0], saturation, vmin=0, vmax=100, dpi=50)
fig.axes[0].set_title(f"{0} yrs", fontsize=32)


def plot_contourf(timevalue: float, mesh: ot.Mesh) -> None:
    fig.axes[0].clear()
    ot.plot.contourf(mesh, saturation, ax=fig.axes[0], dpi=50)
    fig.axes[0].set_title(f"{timevalue:.1f} yrs", fontsize=32)


anim = ot.plot.animate(fig, plot_contourf, ms_r.timevalues, ms_r)

# %% [markdown]
# You can also use any other function to create an animation this way.
# Just make sure, that the function arguments and those passed to the animation
# call fit together.

# %%
ms_x = ms.transform(
    lambda mesh: mesh.sample_over_line([0, 0, 60], [150, 0, 60])
)
fig = ot.plot.line(ms_x[0], ot.variables.saturation)


def plot_line(mesh: ot.Mesh) -> None:
    fig.axes[0].clear()
    ot.plot.line(mesh, saturation, ax=fig.axes[0])
    fig.axes[0].set_ylim([0, 100])
    fig.tight_layout()


anim = ot.plot.animate(fig, plot_line, ms_x)

# %% [markdown]
# The animation can be saved (as mp4) like so:
#
# ..  code-block:: python
#
#   ot.plot.utils.save_animation(anim, "Saturation", fps=5)
#

# sphinx_gallery_start_ignore
# note for developers:
# unfortunately when creating the documentation the animation is run twice:
# once for saving a thumbnail and another time for generating the html_repr
# see .../site-packages/sphinx_gallery/scrapers.py", line 234, 235
# sphinx_gallery_end_ignore
