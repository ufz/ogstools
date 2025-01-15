"""
Workflow: Component-transport model - conversion, simulation, postprocessing
============================================================================
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple mass transport FEFLOW model can be converted to a pyvista.UnstructuredGrid and then
be simulated in OGS with the component transport process.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ogstools as ot
from ogstools.examples import feflow_model_2D_CT_t_560
from ogstools.meshlib import Mesh

ot.plot.setup.show_element_edges = True
# %%
# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_model = ot.FeflowModel(
    feflow_model_2D_CT_t_560, temp_dir / "2D_CT_model"
)
feflow_model.mesh.save(feflow_model.mesh_path)

feflow_concentration = ot.variables.Scalar(
    data_name="single_species_P_CONC",
    output_name="concentration",
    data_unit="mg/l",
    output_unit="mg/l",
)
# The original mesh is clipped to focus on the relevant part of it, where concentration is larger
# than 1e-9 mg/l. The rest of the mesh has concentration values of 0.
ot.plot.contourf(
    feflow_model.mesh.clip_scalar(
        scalars="single_species_P_CONC", invert=False, value=1.0e-9
    ),
    feflow_concentration,
)

# %%
# 2. Save the boundary conditions.
for path, mesh in feflow_model.boundary_conditions.items():
    mesh.save(path)
# %%
# 3. Setup a prj-file to run a OGS-simulation.
feflow_model.setup_prj(
    end_time=int(4.8384e07),
    time_stepping=list(
        zip([10] * 8, [8.64 * 10**i for i in range(8)], strict=False)
    ),
)
prj = feflow_model.project
# The model must be written before it can be run.
prj.write_input()
# Print the prj-file as an example.
ET.dump(ET.parse(feflow_model.mesh_path.with_suffix(".prj")))
# %%
# 4. Run the model.
prj.run_model(logfile=temp_dir / "out.log")
# %%
# 5. Read the results along a line on the upper edge of the mesh parallel to the x-axis and plot them.
ms = ot.MeshSeries(temp_dir / "2D_CT_model.pvd")
# Read the last timestep:
ogs_sim_res = ms.mesh(ms.timesteps[-1])
"""
It is also possible to read the file directly with pyvista:
ogs_sim_res = pv.read(
   temp_dir / "2D_CT_model_ts_65_t_48384000.000000.vtu"
)
"""
profile = np.array([[0.038 + 1.0e-8, 0.005, 0], [0.045, 0.005, 0]])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ogs_sim_res.plot_linesample(
    "dist",
    ot.variables.Scalar(
        data_name="single_species",
        output_name="concentration",
        data_unit="mg/l",
        output_unit="mg/l",
    ),
    profile_points=profile,
    ax=ax,
    resolution=1000,
    grid="major",
    fontsize=18,
    label="OGS",
    color="black",
    linewidth=2,
)
Mesh(feflow_model.mesh).plot_linesample(
    "dist",
    feflow_concentration,
    profile_points=profile,
    ax=ax,
    resolution=1000,
    fontsize=16,
    label="FEFLOW",
    ls=":",
    linewidth=2,
    color="red",
)
ax.legend(loc="best", fontsize=16)
fig.tight_layout()


# %%
# 6. Concentration difference plotted on the mesh.
ogs_sim_res["concentration_difference"] = (
    feflow_model.mesh["single_species_P_CONC"] - ogs_sim_res["single_species"]
)
concentration_difference = ot.variables.Scalar(
    data_name="concentration_difference",
    output_name="concentration",
    data_unit="mg/l",
    output_unit="mg/l",
)

bounds = [0.038, 0.045, 0, 0.01, 0, 0]

ot.plot.contourf(
    ogs_sim_res.clip_box(bounds, invert=False),
    concentration_difference,
)
# %%
# 6.1 Concentration difference plotted along a line.
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ogs_sim_res.plot_linesample(
    "dist",
    concentration_difference,
    profile_points=profile,
    ax=ax,
    resolution=1000,
    grid="both",
    fontsize=18,
    linewidth=2,
    color="green",
    label="Difference FEFLOW-OGS",
)
ax.legend(loc="best", fontsize=16)
fig.tight_layout()
