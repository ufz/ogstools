"""
How to work with FEFLOW data.
=============================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple FEFLOW model consisting of two layers can be converted.
"""

# %%
# 1. Let us convert only the points and cells at first.
from importlib.util import find_spec

# Check for Feflow Python setup
if find_spec("ifm") is None:
    msg = "Could not import ifm. A working Feflow Python setup is required!"
    exit(0)

import ifm_contrib as ifm
import numpy as np

import ogstools.meshplotlib as mpl
from ogstools.fe2vtu import (
    get_geo_mesh,
    update_geo_mesh,
)
from ogstools.propertylib import ScalarProperty

doc = ifm.loadDocument("../../../tests/data/fe2vtu/2layers_model.fem")

mesh = get_geo_mesh(doc)

# sphinx_gallery_start_ignore
# Needed for headless linux systems (CI)
import sys  # noqa: E402

if "linux" in sys.platform:
    import pyvista as pv

    pv.start_xvfb()
# sphinx_gallery_end_ignore
pl = pv.Plotter(off_screen=True)
actor = pl.add_mesh(mesh, show_edges=True)
pl.show()
# mesh.plot(show_edges=True, color=True, off_screen=True)
# %%
# 2. To this mesh we add point and cell data.

mesh = update_geo_mesh(mesh, doc)
mesh.plot(scalars="P_HEAD", off_screen=True)
print(type(mesh))

# %%
# As the converted mesh is a pyvista.UnstructuredGrid, we can apply the MeshPlotLib to it.
mpl.setup.reset()
fig = mpl.plot(mesh, ScalarProperty("P_HEAD"))

# %%
slices = np.reshape(list(mesh.slice_along_axis(n=4, axis="z")), (2, 2))
fig = mpl.plot(slices, ScalarProperty("P_HEAD"))
for ax, slice in zip(fig.axes, np.ravel(slices)):
    ax.set_title(f"z = {slice.center[2]:.1f} {mpl.setup.length.data_unit}")
