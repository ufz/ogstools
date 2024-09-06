"""
Feflowlib: Hydro-thermal model - conversion and simulation
==========================================================
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple hydro thermal FEFLOW model can be converted to a pyvista.UnstructuredGrid and then
be simulated in ot.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pyvista as pv

import ogstools as ot
from ogstools.examples import feflow_model_2D_HT_model
from ogstools.feflowlib import (
    FeflowModel,
)

# %%
# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_model = FeflowModel(
    feflow_model_2D_HT_model, temp_dir / "2D_HT_model.vtu"
)
feflow_temperature = ot.variables.temperature.replace(data_name="P_TEMP")
ot.plot.contourf(feflow_model.mesh, feflow_temperature)


feflow_model.mesh.save(feflow_model.mesh_path)
print(feflow_model.mesh)
# %%
# 2. Extract the boundary conditions.
BC_dict = feflow_model.boundary_conditions
# Since there can be multiple point based boundary conditions on the bulk mesh, they are plotted iteratively.
plotter = pv.Plotter(shape=(len(BC_dict), 1))
for i, (path, boundary_condition) in enumerate(BC_dict.items()):
    # topsurface_2D_HT_model refers to a cell based boundary condition.
    if Path(path).stem != "topsurface_2D_HT_model":
        boundary_condition.save(path)
        plotter.subplot(i, 0)
        plotter.add_mesh(boundary_condition, scalars=Path(path).stem)
        plotter.view_xy()
plotter.show()
# %%
# 3. Setup a prj-file to run a OGS-simulation.
# Get the ogs6py model to create a prj-file and run the simulation.
prj = feflow_model.prj()
# The model must be written before it can be run.
prj.write_input()
# Print the prj-file as an example.
model_prjfile = ET.parse(feflow_model.mesh_path.with_suffix(".prj"))
ET.dump(model_prjfile)
# %%
# 4. Run the model.
prj.run_model(logfile=temp_dir / "out.log")
# %%
# 5. Read the results and plot them.
ms = ot.MeshSeries(temp_dir / "sim_2D_HT_model.pvd")
# Read the last timestep:
ogs_sim_res = ms.mesh(ms.timesteps[-1])
"""
It is also possible to read the file directly with pyvista:
ogs_sim_res = pv.read(
   temp_dir / "sim_2D_HT_model_ts_10_t_100000000000.000000.vtu"
)
"""
# Plot the hydraulic head/height, which was simulated in OGS.
hydraulic_head = ot.variables.Scalar(
    data_name="HEAD_OGS", data_unit="m", output_unit="m"
)
ot.plot.contourf(ogs_sim_res, hydraulic_head)
# %%
# Plot the temperature, which was simulated in OGS.
ot.plot.contourf(ogs_sim_res, ot.variables.temperature)

# %%
# 6. Plot the difference between the FEFLOW and OGS simulation.
feflow_model.mesh["HEAD"] = feflow_model.mesh["P_HEAD"]
ogs_sim_res["HEAD"] = ogs_sim_res["HEAD_OGS"]
# Plot differences in hydraulic head/height.
diff_mesh = ot.meshlib.difference(feflow_model.mesh, ogs_sim_res, "HEAD")
hydraulic_head_diff = ot.variables.Scalar(
    data_name="HEAD_difference", data_unit="m", output_unit="m"
)
ot.plot.contourf(diff_mesh, hydraulic_head_diff)
# %%
feflow_model.mesh["temperature"] = feflow_model.mesh["P_TEMP"]
# Plot differences in temperature.
diff_mesh = ot.meshlib.difference(
    feflow_model.mesh, ogs_sim_res, ot.variables.temperature
)
ot.plot.contourf(diff_mesh, ot.variables.temperature.difference)
