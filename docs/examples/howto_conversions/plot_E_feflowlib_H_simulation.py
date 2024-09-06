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

import ifm_contrib as ifm
import numpy as np
import pyvista as pv

import ogstools as ogs
from ogstools.examples import feflow_model_box_Neumann
from ogstools.feflowlib import (
    convert_properties_mesh,
    extract_cell_boundary_conditions,
    setup_prj_file,
    steady_state_diffusion,
)
from ogstools.feflowlib.tools import (
    extract_point_boundary_conditions,
    get_material_properties,
)

# %%
# 1. Load a FEFLOW model (.fem) as a FEFLOW document, convert and save it. More details on
# how the conversion function works can be found here: :py:mod:`ogstools.feflowlib.convert_properties_mesh`.
feflow_model = ifm.loadDocument(str(feflow_model_box_Neumann))
pyvista_mesh = convert_properties_mesh(feflow_model)

pv.global_theme.colorbar_orientation = "vertical"
pyvista_mesh.plot(
    show_edges=True,
    off_screen=True,
    scalars="P_HEAD",
    cpos=[0, 1, 0.5],
    scalar_bar_args={"position_x": 0.1, "position_y": 0.25},
)
print(pyvista_mesh)
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_mesh_file = temp_dir / "boxNeumann.vtu"
pyvista_mesh.save(feflow_mesh_file)
# %%
# 2. Extract the point conditions (see: :py:mod:`ogstools.feflowlib.extract_point_boundary_conditions`).
point_BC_dict = extract_point_boundary_conditions(temp_dir, pyvista_mesh)
# Since there can be multiple point boundary conditions on the bulk mesh,
# they are saved and plotted iteratively.
plotter = pv.Plotter(shape=(len(point_BC_dict), 1))
for i, (path, boundary_condition) in enumerate(point_BC_dict.items()):
    boundary_condition.save(path)
    plotter.subplot(i, 0)
    plotter.add_mesh(boundary_condition, scalars=Path(path).stem)
plotter.show()
path_topsurface, topsurface = extract_cell_boundary_conditions(
    feflow_mesh_file, pyvista_mesh
)
# On the topsurface can be cell based boundary condition.
# The boundary conditions on the topsurface of the model are required for generalization.
topsurface.save(path_topsurface)
# %%
# 3. Setup a prj-file (see: :py:mod:`ogstools.feflowlib.setup_prj_file`) to run a OGS-simulation.
path_prjfile = feflow_mesh_file.with_suffix(".prj")
prj = ogs.Project(output_file=path_prjfile)
# Get the template prj-file configurations for a steady state diffusion process
ssd_model = steady_state_diffusion(temp_dir / "sim_boxNeumann", prj)
# Include the mesh specific configurations to the template.
model = setup_prj_file(
    bulk_mesh_path=feflow_mesh_file,
    mesh=pyvista_mesh,
    material_properties=get_material_properties(pyvista_mesh, "P_CONDX"),
    process="steady state diffusion",
    model=ssd_model,
)
# The model must be written before it can be run.
model.write_input(path_prjfile)
# Simply print the prj-file as an example.
model_prjfile = ET.parse(path_prjfile)
ET.dump(model_prjfile)
# %%
# 4. Run the model
model.run_model(logfile=temp_dir / "out.log")
# %%
# 5. Read the results and plot them.
ms = ogs.MeshSeries(temp_dir / "sim_boxNeumann.pvd")
# Read the last timestep:
ogs_sim_res = ms.mesh(ms.timesteps[-1])
"""
It is also possible to read the file directly with pyvista:
ogs_sim_res = pv.read(temp_dir / "sim_boxNeumann_ts_1_t_1.000000.vtu")
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
head = ogs.variables.Scalar(
    data_name="HEAD_OGS", data_unit="m", output_unit="m"
)
fig = ogs.plot.contourf(ogs_sim_res.slice(normal="z", origin=[50, 50, 0]), head)


# %%
# 6. Calculate the difference to the FEFLOW simulation and plot it.
diff = pyvista_mesh["P_HEAD"] - ogs_sim_res["HEAD_OGS"]
pyvista_mesh["diff_HEAD"] = diff
pyvista_mesh.plot(
    show_edges=True,
    off_screen=True,
    scalars="diff_HEAD",
    cpos=[0, 1, 0.5],
    scalar_bar_args={"position_x": 0.1, "position_y": 0.25},
)
# %%
# 6.1 Plot the differences in the hydraulic head with :py:mod:`ogstools.plot.contourf`.
# Slices are taken along the z-axis.
diff_head = ogs.variables.Scalar(
    data_name="diff_HEAD", data_unit="m", output_unit="m"
)
slices = np.reshape(list(pyvista_mesh.slice_along_axis(n=4, axis="z")), (2, 2))
fig = ogs.plot.contourf(slices, diff_head)
for ax, slice in zip(fig.axes, np.ravel(slices), strict=False):
    ax.set_title(f"z = {slice.center[2]:.1f} {ms.spatial_output_unit}")

# %%
# Slices are taken along the y-axis.
slices = np.reshape(list(pyvista_mesh.slice_along_axis(n=4, axis="y")), (2, 2))
fig = ogs.plot.contourf(slices, diff_head)
for ax, slice in zip(fig.axes, np.ravel(slices), strict=False):
    ax.set_title(f"y = {slice.center[1]:.1f} {ms.spatial_output_unit}")
