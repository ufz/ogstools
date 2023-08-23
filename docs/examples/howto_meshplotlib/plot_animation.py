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

from ogstools.meshplotlib import examples, setup
from ogstools.meshplotlib.animation import animate
from ogstools.propertylib import ScalarProperty

mesh_series = examples.meshseries_CT_2D
# %%
# Let's use fixed scale limits to prevent rescaling during the animation.
setup.p_min = 0
setup.p_max = 1

# %%
# You can choose which timesteps to render by passing either an int array
# corresponding to the indices, or a float array corresponding to the timevalues
# to render. If a requested timevalue is not part of the timeseries it will be
# interpolated.
timevalues = np.linspace(
    mesh_series.timevalues[0], mesh_series.timevalues[-1], num=25
)

# %%
# Now, let's animate the saturation solution. A timescale at the top
# indicates existing timesteps and the position of the current timevalue.
titles = [f"{tv/(365.25*86400):.1f} yrs" for tv in timevalues]
si = ScalarProperty("Si", "", "%", "Saturation")
anim = animate(mesh_series, si, timevalues, titles)

# sphinx_gallery_start_ignore
# note for developers:
# unfortunately when creating the documentation the animation is run twice:
# once for saving a thumbnail and another time for generating the html_repr
# see .../site-packages/sphinx_gallery/scrapers.py", line 234, 235
# I don't see a way to circumvent this.
# sphinx_gallery_end_ignore
