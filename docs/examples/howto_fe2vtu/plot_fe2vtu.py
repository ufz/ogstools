"""
Conversion of Feflow data to vtk-format and many other things to happen and see.
================================================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

Discribtion of what the converter can do and how it is used and so on.
"""

# %%
# 1. Let us convert only the points and cells of a Feflow model.
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

doc = ifm.loadDocument("../../../ogstools/fe2vtu/tests/test.fem")

mesh = get_geo_mesh(doc)
mesh.plot(show_edges=True, color=True, off_screen=True)

# sphinx_gallery_start_ignore
# Needed for headless linux systems (CI)
import sys  # noqa: E402

if "linux" in sys.platform:
    import pyvista as pv

    pv.start_xvfb()
# sphinx_gallery_end_ignore
mesh.plot(show_edges=True, color=True, off_screen=True)
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
