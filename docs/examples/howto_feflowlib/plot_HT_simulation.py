"""
How to convert a hydro thermal FEFLOW model and simulate it in OGS.
===================================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple hydro thermal FEFLOW model can be converted to a pyvista.UnstructuredGrid and then
be simulated in OGS.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import ifm_contrib as ifm
import pyvista as pv
from ogs6py import ogs

import ogstools.meshplotlib as mpl
from ogstools.feflowlib import (
    convert_properties_mesh,
    hydro_thermal,
    setup_prj_file,
)
from ogstools.feflowlib.examples import path_2D_HT_model
from ogstools.feflowlib.tools import (
    extract_point_boundary_conditions,
    get_material_properties_of_HT_model,
)
from ogstools.meshlib import difference
from ogstools.propertylib import presets

# %%
# 1. Load a FEFLOW model (.fem) as a FEFLOW document, convert and save it.
feflow_model = ifm.loadDocument(path_2D_HT_model)
feflow_pv_mesh = convert_properties_mesh(feflow_model)
presets.temperature.data_name = "P_TEMP"
mpl.plot(feflow_pv_mesh, presets.temperature)

path_writing = Path(tempfile.mkdtemp("feflow_test_simulation"))
path_mesh = path_writing / "2D_HT_model.vtu"
feflow_pv_mesh.save(path_mesh)
print(feflow_pv_mesh)
# %%
# 2. Extract the point conditions.

point_BC_dict = extract_point_boundary_conditions(path_writing, feflow_pv_mesh)
# Since there can be multiple point boundary conditions on the bulk mesh, they are plotted iteratively.
plotter = pv.Plotter(shape=(len(point_BC_dict), 1))
for i, (path, boundary_condition) in enumerate(point_BC_dict.items()):
    boundary_condition.save(path)
    plotter.subplot(i, 0)
    plotter.add_mesh(boundary_condition, scalars=Path(path).stem)
    plotter.view_xy()
plotter.show()
# %%
# 3. Setup a prj-file to run a OGS-simulation
path_prjfile = str(path_mesh.with_suffix(".prj"))
prjfile = ogs.OGS(PROJECT_FILE=str(path_prjfile))
# Get the template prj-file configurations for a hydro thermal process.
HT_model = hydro_thermal(
    path_writing / "sim_2D_HT_model",
    prjfile,
    True,
)
# Include the mesh specific configurations to the template.
model = setup_prj_file(
    path_mesh,
    feflow_pv_mesh,
    get_material_properties_of_HT_model(feflow_pv_mesh),
    "hydro thermal",
    model=HT_model,
)
# The model must be written before it can be run.
model.write_input(str(path_prjfile))
# Print the prj-file as an example.
read_model = ET.parse(str(path_prjfile))
root = read_model.getroot()
ET.dump(root)
# %%
# 4. Run the model.
model.run_model(logfile=str(path_writing / "out.log"))
# %%
# 5. Read the results and plot them.
ogs_sim_res = pv.read(
    str(path_writing / "sim_2D_HT_model_ts_10_t_100000000000.000000.vtu")
)
# Plot the hydraulic head/height, which was simulated in OGS.
presets.hydraulic_height.data_name = "HEAD_OGS"
mpl.plot(ogs_sim_res, presets.hydraulic_height)
# %%
# Plot the temperature, which was simulated in OGS.
presets.temperature.data_name = "temperature"
mpl.plot(ogs_sim_res, presets.temperature)

# %%
# 6. Plot the difference between the FEFLOW and OGS simulation.
feflow_pv_mesh["HEAD"] = feflow_pv_mesh["P_HEAD"]
ogs_sim_res["HEAD"] = ogs_sim_res["HEAD_OGS"]
presets.hydraulic_height.data_name = "HEAD"
# Plot differences in hydraulic head/height.
mpl.plot(
    difference(feflow_pv_mesh, ogs_sim_res, presets.hydraulic_height),
    presets.hydraulic_height,
)
# %%
feflow_pv_mesh["temperature"] = feflow_pv_mesh["P_TEMP"]
# Plot differences in temperature.
mpl.plot(
    difference(feflow_pv_mesh, ogs_sim_res, presets.temperature),
    presets.temperature,
)
