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

from ogstools import examples
from ogstools.meshplotlib import setup
from ogstools.meshplotlib.animation import animate
from ogstools.propertylib import Scalar

setup.reset()
mesh_series = examples.load_meshseries_CT_2D_XDMF()
mesh_property = Scalar(
    data_name="Si", data_unit="", output_unit="%", output_name="Saturation"
)

# %% [markdown]
# To read your own data as a mesh series you can do:
#
# ..  code-block:: python
#
#   from ogstools.meshlib import MeshSeries
#   mesh_series = MeshSeries("filepath/filename_pvd_or_xdmf")
#
# You can also use a property from the available presets instead of needing to
# create your own:
# :ref:`sphx_glr_auto_examples_howto_propertylib_plot_propertylib.py`

# %% [markdown]
# Let's use fixed scale limits to prevent rescaling during the animation.

# %%
setup.p_min = 0
setup.p_max = 100
setup.dpi = 50

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
# a really long time.

# %%
titles = [f"{tv/(365.25*86400):.1f} yrs" for tv in timevalues]
anim = animate(mesh_series, mesh_property, timevalues, titles)

# %% [markdown]
# The animation can be saved (as mp4) like so:
#
# ..  code-block:: python
#
#   from ogstools.meshplotlib.animation import save_animation
#   save_animation(anim, "Saturation", fps=5)
#

# sphinx_gallery_start_ignore
# note for developers:
# unfortunately when creating the documentation the animation is run twice:
# once for saving a thumbnail and another time for generating the html_repr
# see .../site-packages/sphinx_gallery/scrapers.py", line 234, 235
# sphinx_gallery_end_ignore
