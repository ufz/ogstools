"""
How to work with FEFLOW data in pyvista.
========================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple FEFLOW model consisting of two layers can be converted.
"""

# %%
# 1. Let us convert only the points and cells at first.
import ifm_contrib as ifm

import ogstools.meshplotlib as mpl
from ogstools.fe2vtu import (
    get_geo_mesh,
    update_geo_mesh,
)
from ogstools.propertylib import ScalarProperty

doc = ifm.loadDocument("../../../tests/data/fe2vtu/2layers_model.fem")
mesh = get_geo_mesh(doc)
mesh.plot(show_edges=True, off_screen=True)
# %%
# 2. To this mesh we add point and cell data.
mesh = update_geo_mesh(mesh, doc)
mesh.plot(scalars="P_HEAD", show_edges=True, off_screen=True)
print(mesh)
# %%
# 3. As the FEFLOW data now are a pyvista.UnstructuredGrid, all pyvista functionalities can be applied to it.
# Further information can be found at https://docs.pyvista.org/version/stable/user-guide/simple.html.
# For example it can be saved as a VTK Unstructured Grid File (\*.vtu).
# This allows to use the FEFLOW model for ``OGS`` simulation or to observe it in ``Paraview```.
mesh.save("2layers_model.vtu")
# %%
# 4. As the converted mesh is a pyvista.UnstructuredGrid, we can plot it using meshplotlib.
mpl.plot(mesh, ScalarProperty("P_HEAD"))
