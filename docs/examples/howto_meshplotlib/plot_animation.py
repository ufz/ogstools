"""
How to create Animations
========================

.. sectionauthor:: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)

For this example we load a 2D meshseries dataset from within the ``meshplotlib`` examples.
...
"""
# %%
import numpy as np

from ogstools.meshplotlib import examples, setup
from ogstools.meshplotlib.animation import animate
from ogstools.propertylib.defaults import temperature

setup.p_max = 48.5
setup.num_levels = 25
# TODO: needs better example
mesh_series = examples.meshseries_THM_2D

timevalues = np.linspace(0, mesh_series.timevalues[-1], num=5)
titles = [f"{tv/(365.25*86400):.1f} yrs" for tv in timevalues]
anim = animate(mesh_series, temperature, timevalues, titles)
# %%
