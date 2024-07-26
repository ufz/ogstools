"""
Feflowlib: How to work with FEFLOW data in pyvista.
===================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple FEFLOW model consisting of two layers can be converted to a pyvista.UnstructuredGrid.
"""

# %%
# 0. Necessary imports
import tempfile
from pathlib import Path

import ogstools as ot
from ogstools.examples import feflow_model_2layers
from ogstools.feflowlib import feflowModel

# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_model = feflowModel(feflow_model_2layers, temp_dir / "2layers.vtu")
feflow_model.mesh.plot(scalars="P_HEAD", show_edges=True, off_screen=True)
# Print information about the mesh.
print(feflow_model.mesh)
# %%
# 2. As the FEFLOW data now are a pyvista.UnstructuredGrid, all pyvista functionalities can be applied to it.
# Further information can be found at https://docs.pyvista.org/version/stable/user-guide/simple.html.
# For example it can be saved as a VTK Unstructured Grid File (\*.vtu).
# This allows to use the FEFLOW model for ``OGS`` simulation or to observe it in ``Paraview```.
feflow_model.mesh.save("2layers_model.vtu")
# %%
# 4. Use the ogstools plotting functionalities.
fig = ot.plot.contourf(pv_mesh.slice("z"), "P_HEAD")
