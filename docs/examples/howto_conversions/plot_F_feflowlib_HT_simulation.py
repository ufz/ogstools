"""
Workflow: Hydro-thermal model - conversion, simulation and post-processing
==========================================================================
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple hydro thermal FEFLOW model can be converted to a pyvista.UnstructuredGrid and then
be simulated in ot.
"""

# %%
# 0. Necessary imports
import tempfile
from pathlib import Path

import pyvista as pv

import ogstools as ot
from ogstools.examples import feflow_model_2D_HT

# %%
# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_model = ot.FeflowModel(feflow_model_2D_HT, temp_dir / "2D_HT_model.vtu")
# %%
# 2. Plot the temperature simulated in FEFLOW on the mesh, and print information about the mesh.
feflow_temperature = ot.variables.temperature.replace(data_name="P_TEMP")
ot.plot.contourf(feflow_model.mesh, feflow_temperature)
print(feflow_model.mesh)
# %%
# 3. Extract the subdomains.
subdomains = feflow_model.subdomains
# Since there can be multiple boundary conditions in the subdomains, they are plotted iteratively.
plotter = pv.Plotter(shape=(len(subdomains), 1))
for i, (name, boundary_condition) in enumerate(subdomains.items()):
    # topsurface refers to a cell based boundary condition.
    if name != "topsurface":
        plotter.subplot(i, 0)
        plotter.add_mesh(boundary_condition, scalars=name)
        plotter.view_xy()
plotter.show()
# %%
# 4. Setup a prj-file to run a OGS-simulation.
# Get the ogs6py model to create a prj-file and run the simulation.
feflow_model.setup_prj(end_time=1e11, time_stepping=[(1, 1e10)])
# %%
# 5. Run the model.
feflow_model.run()
# %%
# 6. Read the results and plot them.
ms = ot.MeshSeries(temp_dir / "2D_HT_model.pvd")
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
# 7. Plot the difference between the FEFLOW and OGS simulation.
feflow_model.mesh["HEAD"] = feflow_model.mesh["P_HEAD"]
ogs_sim_res["HEAD"] = ogs_sim_res["HEAD_OGS"]
# Plot differences in hydraulic head/height.
diff_mesh = ot.meshlib.difference(feflow_model.mesh, ogs_sim_res, "HEAD")
hydraulic_head_diff = ot.variables.Scalar(
    data_name="HEAD_difference", data_unit="m", output_unit="m"
)
ot.plot.contourf(diff_mesh, hydraulic_head_diff, vmin=-1.5e-9, vmax=1.5e-9)
# %%
# Plot differences in temperature.
feflow_model.mesh["temperature"] = feflow_model.mesh["P_TEMP"]
# Plot differences in temperature.
diff_mesh = ot.meshlib.difference(
    feflow_model.mesh, ogs_sim_res, ot.variables.temperature
)
ot.plot.contourf(
    diff_mesh, ot.variables.temperature.difference, vmin=-8.7e-9, vmax=8.7e-9
)
