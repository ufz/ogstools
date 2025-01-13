"""
Feflowlib: How to convert the mesh of a FEFLOW model and modify it afterwards.
==============================================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

This example shows how the mesh of a FEFLOW model can be converted and modified after conversion.
"""

# %%
# 0. Necessary imports
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv

import ogstools as ogs
from ogstools.examples import feflow_model_2D_HT
from ogstools.feflowlib import FeflowModel
from ogstools.feflowlib._tools import assign_bulk_ids

# %%
# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("converted_models"))
feflow_model = FeflowModel(feflow_model_2D_HT, temp_dir / "HT_model")
mesh = feflow_model.mesh
# Print information about the mesh.
print(mesh)
# %%
# 2. Plot the temperature simulated in FEFLOW on the mesh.
fig = ogs.plot.contourf(mesh, "P_TEMP", show_edges=True)
# %%
# 3. As the FEFLOW data now are a pyvista.UnstructuredGrid, all pyvista functionalities can be applied to it.
# Further information can be found at https://docs.pyvista.org/version/stable/user-guide/simple.html.
# For example it can be saved as a VTK Unstructured Grid File (\*.vtu).
# This allows to use the FEFLOW model for ``OGS`` simulation or to observe it in ``Paraview```.
mesh.save(temp_dir / "HT_mesh.vtu")
# %%
# 4. Run the model.
feflow_model.set_up_prj(end_time=1e11, time_stepping=[(1, 1e10)])
# feflow_model.run()
# ms = ogs.MeshSeries(temp_dir / "HT_model.pvd")
# Read the last timestep:
# ogs_sim_res = ms.mesh(ms.timesteps[-1])
# ogs.plot.contourf(ogs_sim_res, ogs.variables.temperature)
# %%
# 5. The boundary meshes are manipulated to alter the model.
boundary_conditions = feflow_model.boundary_conditions
feflow_model.save()
# 5.1 The Dirichlet boundary conditions for the hydraulic head are set to 0.
bc_flow = boundary_conditions[str(temp_dir / "P_BC_FLOW.vtu")]["P_BC_FLOW"]
boundary_conditions[str(temp_dir / "P_BC_FLOW.vtu")]["P_BC_FLOW"][
    bc_flow == 0
] = 0
boundary_conditions[str(temp_dir / "P_BC_FLOW.vtu")]["P_BC_FLOW"][
    bc_flow == 10
] = 0
for path, boundary_mesh in boundary_conditions.items():
    boundary_mesh.save(path)
prj = feflow_model.project
prj.run_model()
# %%
# 5.2 The corresponding simulation results look like.
ms = ogs.MeshSeries(temp_dir / "HT_model.pvd")
# Read the last timestep:
ogs_sim_res = ms.mesh(ms.timesteps[-1])
ogs.plot.contourf(ogs_sim_res, ogs.variables.temperature)
# %%
# 5.3 To add new points to the existing boundary mesh of the Dirichilet temperature
# boundary conditions, we need the following steps:
assign_bulk_ids(mesh)
# Get the wanted points of the bulk mesh by their ids.
wanted_pts = [1492, 1482, 1481, 1491, 1479, 1480, 1839, 1840]
pts_to_add = mesh.extract_points(
    [pt in wanted_pts for pt in mesh["bulk_node_ids"]],
    adjacent_cells=False,
    include_cells=False,
)
pts_to_add["bulk_node_ids"] = wanted_pts
# Define the temperature values of these pts
pts_to_add["P_BC_HEAT"] = [300] * len(wanted_pts)
# Merge these pts with the existing BC-mesh.
new_bc = pv.merge(
    [boundary_conditions[str(temp_dir / "P_BC_HEAT.vtu")], pts_to_add],
    main_has_priority=False,
)
# Define the temperature values
heat_val = np.append(
    np.array(boundary_conditions[str(temp_dir / "P_BC_HEAT.vtu")]["P_BC_HEAT"]),
    np.array(pts_to_add["P_BC_HEAT"]),
)
new_bc["P_BC_HEAT"] = heat_val
# Define the bulk node ids.
ids = np.array(
    np.append(
        np.array(
            boundary_conditions[str(temp_dir / "P_BC_HEAT.vtu")][
                "bulk_node_ids"
            ]
        ),
        np.array(pts_to_add["bulk_node_ids"]),
    ),
    dtype=np.uint64,
)
new_bc["bulk_node_ids"] = ids
# Overwrite the old boundary mesh.
new_bc.save(str(temp_dir / "P_BC_HEAT.vtu"))
# %%
# 6. Run the new model and plot the simulation results.
prj = feflow_model.project
prj.run_model()
ms = ogs.MeshSeries(temp_dir / "HT_model.pvd")
ogs_sim_res = ms.mesh(ms.timesteps[-1])
ogs.plot.contourf(ogs_sim_res, ogs.variables.temperature)
