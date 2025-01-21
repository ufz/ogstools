"""
How to create Animations
========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

To demonstrate the creation of an animated plot we use a component transport
example from the ogs benchmark gallery
(https://www.opengeosys.org/docs/benchmarks/hydro-component/elder/).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import ogstools as ot
from ogstools import examples

mesh_series = examples.load_meshseries_CT_2D_XDMF()

# %% [markdown]
# To read your own data as a mesh series you can do:
#
# ..  code-block:: python
#
#   from ogstools.meshlib import MeshSeries
#   mesh_series = MeshSeries("filepath/filename_pvd_or_xdmf")

# %% [markdown]
# You can choose which timesteps to render by passing either an int array
# corresponding to the indices, or a float array corresponding to the timevalues
# to render. If a requested timevalue is not part of the timeseries it will be
# interpolated. In this case every second frame will be interpolated.

# %%
timevalues = np.linspace(
    mesh_series.timevalues[0], mesh_series.timevalues[-1], num=25
)

# %% [markdown]
# Now, let's animate the saturation solution. A timescale at the top
# indicates existing timesteps and the position of the current timevalue.
# Note that rendering many frames in conjunction with large meshes might take
# a really long time. We can pass a ``plot_func`` which can apply custom
# formatting and / or plotting. To modify the domain, we can use the transform
# method of MeshSeries.


def mesh_func(mesh: ot.Mesh) -> ot.Mesh:
    "Clip the left half of the mesh."
    return mesh.clip("-x", [0, 0, 0])


def plot_func(ax: plt.Axes, timevalue: float) -> None:
    "Add the time to the title."
    ax.set_title(f"{timevalue/(365.25*86400):.1f} yrs")


# %%
anim = mesh_series.transform(mesh_func).animate(
    ot.variables.saturation, timevalues, plot_func, vmin=0, vmax=100, dpi=50
)

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
