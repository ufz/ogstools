"""
Feflowlib: Hydro-thermal model - conversion and simulation
==========================================================
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

import ogstools as ogs
from ogstools.examples import feflow_model_2D_HT_model
from ogstools.feflowlib import (
    convert_properties_mesh,
    hydro_thermal,
    setup_prj_file,
)
from ogstools.feflowlib.tools import (
    extract_point_boundary_conditions,
    get_material_properties_of_HT_model,
)

# %%
# 1. Load a FEFLOW model (.fem) as a FEFLOW document, convert and save it. More details on
# how the conversion function works can be found here: :py:mod:`ogstools.feflowlib.convert_properties_mesh`.
feflow_model = ifm.loadDocument(str(feflow_model_2D_HT_model))
feflow_pv_mesh = convert_properties_mesh(feflow_model)
feflow_temperature = ogs.variables.temperature.replace(data_name="P_TEMP")
ogs.plot.contourf(feflow_pv_mesh, feflow_temperature)

temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_mesh_file = temp_dir / "2D_HT_model.vtu"
feflow_pv_mesh.save(feflow_mesh_file)
print(feflow_pv_mesh)
# %%
# 2. Extract the point conditions (see: :py:mod:`ogstools.feflowlib.extract_point_boundary_conditions`).
point_BC_dict = extract_point_boundary_conditions(temp_dir, feflow_pv_mesh)
# Since there can be multiple point boundary conditions on the bulk mesh, they are plotted iteratively.
plotter = pv.Plotter(shape=(len(point_BC_dict), 1))
for i, (path, boundary_condition) in enumerate(point_BC_dict.items()):
    boundary_condition.save(path)
    plotter.subplot(i, 0)
    plotter.add_mesh(boundary_condition, scalars=Path(path).stem)
    plotter.view_xy()
plotter.show()
# %%
# 3. Setup a prj-file (see: :py:mod:`ogstools.feflowlib.setup_prj_file`) to run a OGS-simulation.
path_prjfile = feflow_mesh_file.with_suffix(".prj")
prj = ogs.Project(output_file=path_prjfile)
# Get the template prj-file configurations for a hydro thermal process.
HT_model = hydro_thermal(temp_dir / "sim_2D_HT_model", prj, dimension=2)
# Include the mesh specific configurations to the template.
model = setup_prj_file(
    bulk_mesh_path=feflow_mesh_file,
    mesh=feflow_pv_mesh,
    material_properties=get_material_properties_of_HT_model(feflow_pv_mesh),
    process="hydro thermal",
    model=HT_model,
)
# The model must be written before it can be run.
model.write_input(path_prjfile)
# Print the prj-file as an example.
model_prjfile = ET.parse(path_prjfile)
ET.dump(model_prjfile)
# %%
# 4. Run the model.
model.run_model(logfile=temp_dir / "out.log")
# %%
# 5. Read the results and plot them.
ms = ogs.MeshSeries(temp_dir / "sim_2D_HT_model.pvd")
# Read the last timestep:
ogs_sim_res = ms.mesh(ms.timesteps[-1])
"""
It is also possible to read the file directly with pyvista:
ogs_sim_res = pv.read(
   temp_dir / "sim_2D_HT_model_ts_10_t_100000000000.000000.vtu"
)
"""
# Plot the hydraulic head/height, which was simulated in OGS.
hydraulic_head = ogs.variables.Scalar(
    data_name="HEAD_OGS", data_unit="m", output_unit="m"
)
ogs.plot.contourf(ogs_sim_res, hydraulic_head)
# %%
# Plot the temperature, which was simulated in OGS.
ogs.plot.contourf(ogs_sim_res, ogs.variables.temperature)

# %%
# 6. Plot the difference between the FEFLOW and OGS simulation.
feflow_pv_mesh["HEAD"] = feflow_pv_mesh["P_HEAD"]
ogs_sim_res["HEAD"] = ogs_sim_res["HEAD_OGS"]
# Plot differences in hydraulic head/height.
diff_mesh = ogs.meshlib.difference(feflow_pv_mesh, ogs_sim_res, "HEAD")
hydraulic_head_diff = ogs.variables.Scalar(
    data_name="HEAD_difference", data_unit="m", output_unit="m"
)
ogs.plot.contourf(diff_mesh, hydraulic_head_diff)
# %%
feflow_pv_mesh["temperature"] = feflow_pv_mesh["P_TEMP"]
# Plot differences in temperature.
diff_mesh = ogs.meshlib.difference(
    feflow_pv_mesh, ogs_sim_res, ogs.variables.temperature
)
ogs.plot.contourf(diff_mesh, ogs.variables.temperature.difference)
