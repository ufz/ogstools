"""
How to convert a FEFLOW model and simulate it in OGS.
=====================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple FEFLOW model can be converted to a pyvista.UnstructuredGrid and then
be simulated in OGS.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import ifm_contrib as ifm
from ogs6py import ogs
from pyvista import read

from ogstools.feflowlib import (
    convert_properties_mesh,
    extract_cell_boundary_conditions,
    setup_prj_file,
    steady_state_diffusion,
)
from ogstools.feflowlib.examples import path_box_Neumann
from ogstools.feflowlib.tools import (
    extract_point_boundary_conditions,
    get_material_properties,
)

# %%
# 1. Load a FEFLOW model (.fem) as a FEFLOW document and convert it.
feflow_model = ifm.loadDocument(path_box_Neumann)
pyvista_mesh = convert_properties_mesh(feflow_model)
pyvista_mesh.plot(
    show_edges=True, off_screen=True, scalars="P_HEAD", cpos=[0, 1, 0.5]
)
print(pyvista_mesh)
path_writing = Path(tempfile.mkdtemp("feflow_test_simulation"))
path_mesh = path_writing / "boxNeumann.vtu"
pyvista_mesh.save(str(path_mesh))
# %%
# 2. Extract the point and cell boundary conditions and write them to a temporary directory.
point_BC_dict = extract_point_boundary_conditions(path_writing, pyvista_mesh)
# Since there can be multiple point boundary conditions on the bulk mesh,
# they are saved and plotted iteratively.
for path, boundary_condition in point_BC_dict.items():
    boundary_condition.save(path)
    boundary_condition.plot()
path_topsurface, topsurface = extract_cell_boundary_conditions(
    path_mesh, pyvista_mesh
)
# On the topsurface can be cell based boundary condition.
# The boundary conditions on the topsurface of the model are required for generalization.
topsurface.save(path_topsurface)
# %%
# 3. Setup a prj-file to run a OGS-simulation
path_prjfile = str(path_mesh.with_suffix(".prj"))
prjfile = ogs.OGS(PROJECT_FILE=str(path_prjfile))
# Get the template prj-file configurations for a steady state diffusion process
model = steady_state_diffusion(
    path_writing / "sim_boxNeumann",
    prjfile,
)
# Include the mesh specific configurations to the template.
model = setup_prj_file(
    path_mesh,
    pyvista_mesh,
    get_material_properties(pyvista_mesh, "P_CONDX"),
    "steady state diffusion",
    model,
)
# The model must be written before it can be run.
model.write_input(str(path_prjfile))
# Simply print the prj-file as an example.
read_model = ET.parse(str(path_prjfile))
root = read_model.getroot()
ET.dump(root)
# %%
# 4. Run the model
model.run_model(logfile=str(path_writing / "out.log"))
# %%
# 5. Read the results and plot them.
ogs_sim_res = read(str(path_writing / "sim_boxNeumann_ts_1_t_1.000000.vtu"))
ogs_sim_res.plot(
    show_edges=True, off_screen=True, scalars="HEAD_OGS", cpos=[0, 1, 0.5]
)

# %%
# 6. Calculate the difference to the FEFLOW simulation and plot it.
diff = pyvista_mesh["P_HEAD"] - ogs_sim_res["HEAD_OGS"]
pyvista_mesh["diff_HEAD"] = diff
pyvista_mesh.plot(
    show_edges=True, off_screen=True, scalars="diff_HEAD", cpos=[0, 1, 0.5]
)
# %%
