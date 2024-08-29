"""
Feflowlib: Hydraulic model - conversion and simulation
======================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple flow/hydraulic FEFLOW model can be converted to a pyvista.UnstructuredGrid and then
be simulated in OGS.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv

import ogstools as ot
from ogstools.examples import feflow_model_box_Neumann
from ogstools.feflowlib import (
    FeflowModel,
)

# %%
# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_model = FeflowModel(feflow_model_box_Neumann, temp_dir / "boxNeumann")

pv.global_theme.colorbar_orientation = "vertical"
feflow_model.mesh.plot(
    show_edges=True,
    off_screen=True,
    scalars="P_HEAD",
    cpos=[0, 1, 0.5],
    scalar_bar_args={"position_x": 0.1, "position_y": 0.25},
)
print(feflow_model.mesh)
feflow_model.mesh.save(feflow_model.mesh_path)
# %%
# 2. Extract the boundary conditions.
BC_dict = feflow_model.boundary_conditions
# Since there can be multiple point based boundary conditions on the bulk mesh,
# they are saved and plotted iteratively.
plotter = pv.Plotter(shape=(len(BC_dict), 1))
for i, (path, boundary_condition) in enumerate(BC_dict.items()):
    # topsurface_boxNeumann refers to a cell based boundary condition.
    if Path(path).stem != "topsurface_boxNeumann":
        boundary_condition.save(path)
        plotter.subplot(i, 0)
        plotter.add_mesh(boundary_condition, scalars=Path(path).stem)
plotter.show()
path_topsurface, topsurface = list(BC_dict.items())[-1]
# On the topsurface are cell based boundary condition.
topsurface.save(path_topsurface)
# %%
# 3. Setup a prj-file to run a OGS-simulation.
model = feflow_model.prj()
# The model must be written before it can be run.
model.write_input()
# Print the prj-file as an example.
model_prjfile = ET.parse(feflow_model.mesh_path.with_suffix(".prj"))
ET.dump(model_prjfile)
# %%
# 4. Run the model
model.run_model(logfile=temp_dir / "out.log")
# %%
# 5. Read the results and plot them.
ms = ot.MeshSeries(temp_dir / "sim_boxNeumann.pvd")
# Read the last timestep:
ogs_sim_res = ms.mesh(ms.timesteps[-1])
"""
It is also possible to read the file directly with pyvista:
ogs_sim_res = pv.read(temp_dir / "boxNeumann_ts_1_t_1.000000.vtu")
"""
ogs_sim_res.plot(
    show_edges=True,
    off_screen=True,
    scalars="HEAD_OGS",
    cpos=[0, 1, 0.5],
    scalar_bar_args={"position_x": 0.1, "position_y": 0.25},
)
# %%
# 5.1 Plot the hydraulic head simulated in OGS with :py:mod:`ogstools.plot.contourf`.
head = ot.variables.Scalar(data_name="HEAD_OGS", data_unit="m", output_unit="m")
fig = ot.plot.contourf(ogs_sim_res.slice(normal="z", origin=[50, 50, 0]), head)


# %%
# 6. Calculate the difference to the FEFLOW simulation and plot it.
diff = feflow_model.mesh["P_HEAD"] - ogs_sim_res["HEAD_OGS"]
feflow_model.mesh["diff_HEAD"] = diff
feflow_model.mesh.plot(
    show_edges=True,
    off_screen=True,
    scalars="diff_HEAD",
    cpos=[0, 1, 0.5],
    scalar_bar_args={"position_x": 0.1, "position_y": 0.25},
)
# %%
# 6.1 Plot the differences in the hydraulic head with :py:mod:`ogstools.plot.contourf`.
# Slices are taken along the z-axis.
diff_head = ot.variables.Scalar(
    data_name="diff_HEAD", data_unit="m", output_unit="m"
)
slices = np.reshape(
    list(feflow_model.mesh.slice_along_axis(n=4, axis="z")), (2, 2)
)
fig = ot.plot.contourf(slices, diff_head)
for ax, slice in zip(fig.axes, np.ravel(slices), strict=False):
    ax.set_title(f"z = {slice.center[2]:.1f} {ms.spatial_output_unit}")

# %%
# Slices are taken along the y-axis.
slices = np.reshape(
    list(feflow_model.mesh.slice_along_axis(n=4, axis="y")), (2, 2)
)
fig = ot.plot.contourf(slices, diff_head)
for ax, slice in zip(fig.axes, np.ravel(slices), strict=False):
    ax.set_title(f"y = {slice.center[1]:.1f} {ms.spatial_output_unit}")
