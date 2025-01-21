"""
Workflow: Component-transport model - conversion, simulation, postprocessing
============================================================================
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple mass transport FEFLOW model can be
converted to a pyvista.UnstructuredGrid and then be simulated in OGS with the
component transport process.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt

import ogstools as ot
from ogstools.examples import feflow_model_2D_CT_t_560

ot.plot.setup.show_element_edges = True

# %%
# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_model = ot.FeflowModel(
    feflow_model_2D_CT_t_560, temp_dir / "2D_CT_model"
)
# name the feflow concentratiob result the same as in OGS for easier comparison
feflow_model.mesh["single_species"] = feflow_model.mesh["single_species_P_CONC"]
concentration = ot.variables.Scalar(
    data_name="single_species", output_name="concentration",
    data_unit="mg/l", output_unit="mg/l",
)  # fmt: skip
# The original mesh is clipped to focus on the relevant part of it, where
# concentration is larger than 1e-9 mg/l. The rest of the mesh has concentration
# values of 0.
clipped_mesh = feflow_model.mesh.clip_scalar(
    scalars="single_species", invert=False, value=1.0e-9
)
ot.plot.contourf(clipped_mesh, concentration)
# %%
# 2. Setup a prj-file to run a OGS-simulation.
time_steps = list(
    zip([10] * 8, [8.64 * 10**i for i in range(8)], strict=False)
)
feflow_model.setup_prj(end_time=int(4.8384e07), time_stepping=time_steps)
# Save the model (mesh, subdomains and project file).
feflow_model.save()
# Print the prj-file as an example.
ET.dump(ET.parse(feflow_model.mesh_path.with_suffix(".prj")))
# %%
# 3. Run the model.
feflow_model.run()
# %%
# 4. Read the last timestep and plot the results along a line on the upper edge
# of the mesh parallel to the x-axis.
ogs_sim_res = ot.MeshSeries(temp_dir / "2D_CT_model.pvd")[-1]
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
pts = [[0.038 + 1.0e-8, 0.005, 0], [0.045, 0.005, 0]]
for i, mesh in enumerate([ogs_sim_res, feflow_model.mesh]):
    sample = mesh.sample_over_line(*pts)
    label = ["OGS", "FEFLOW"][i]
    ot.plot.line(
        sample, concentration, ax=ax, color="kr"[i], label=label, ls="-:"[i]
    )
fig.tight_layout()
# %%
# 5. Concentration difference plotted on the mesh.

diff = ot.meshlib.difference(feflow_model.mesh, ogs_sim_res, concentration)
diff_clipped = diff.clip_box([0.038, 0.045, 0, 0.01, 0, 0], invert=False)
fig = ot.plot.contourf(diff_clipped, concentration.difference, fontsize=20)
# %%
# 5.1 Concentration difference plotted along a line.
diff_sample = diff.sample_over_line(*pts)
fig = ot.plot.line(
    diff_sample, concentration.difference, label="Difference FEFLOW-OGS"
)
fig.tight_layout()
