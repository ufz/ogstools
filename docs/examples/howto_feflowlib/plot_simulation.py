"""
How to work with FEFLOW data in pyvista.
========================================

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
pv_mesh = convert_properties_mesh(feflow_model)
pv_mesh.plot(
    show_edges=True, off_screen=True, scalars="P_HEAD", cpos=[0, 1, 0.5]
)
print(pv_mesh)
path_writing = Path(tempfile.mkdtemp("feflow_test_simulation"))
pv_mesh.save(str(path_writing / "boxNeumann.vtu"))
# %%
# 2. Extract the point and cell boundary conditions and write them to a temporary directory.
point_BC_dict = extract_point_boundary_conditions(path_writing, pv_mesh)
# Since there can be multiple point boundary conditions on the bulk mesh,
# they are saved and plotted iteratively.
for path, boundary_condition in point_BC_dict.items():
    boundary_condition.save(path)
    boundary_condition.plot()
topsurface = extract_cell_boundary_conditions(
    path_writing / "boxNeumann.vtu", pv_mesh
)
topsurface[1].save(topsurface[0])
topsurface[1].plot()
# %%
# 3. Setup a prj-file to run a OGS-simulation
prjfile = str(path_writing / "boxNeumann_test.prj")
# Get the template prj-file configurations for a steady state diffusion process
model = steady_state_diffusion(
    str(path_writing / "sim_boxNeumann"),
    ogs.OGS(PROJECT_FILE=prjfile),
)
# Include the mesh specific configurations to the template.
model = setup_prj_file(
    path_writing / "boxNeumann.vtu",
    pv_mesh,
    get_material_properties(pv_mesh, "P_CONDX"),
    "steady state diffusion",
    model,
)
# The model must be written before it can be run.
model.write_input(prjfile)
# Simply print the prj-file as an example.
read_model = ET.parse(prjfile)
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
