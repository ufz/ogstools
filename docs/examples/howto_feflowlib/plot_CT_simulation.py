"""
Component-transport model - conversion and simulation
=====================================================
.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

In this example we show how a simple mass transport FEFLOW model can be converted to a pyvista.UnstructuredGrid and then
be simulated in OGS with the component transport process.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import ifm_contrib as ifm
import matplotlib.pyplot as plt
from ogs6py import ogs

import ogstools.meshplotlib as mpl
from ogstools.examples import feflow_model_2D_CT_t_560
from ogstools.feflowlib import (
    component_transport,
    convert_properties_mesh,
    get_material_properties_of_CT_model,
    get_species,
    setup_prj_file,
    write_point_boundary_conditions,
)
from ogstools.meshlib import MeshSeries
from ogstools.propertylib import Scalar

# %%
# 1. Load a FEFLOW model (.fem) as a FEFLOW document, convert and save it. More details on
# how the conversion function works can be found here: :py:mod:`ogstools.feflowlib.convert_properties_mesh`.
feflow_model = ifm.loadDocument(str(feflow_model_2D_CT_t_560))
feflow_pv_mesh = convert_properties_mesh(feflow_model)

temp_dir = Path(tempfile.mkdtemp("feflow_test_simulation"))
feflow_mesh_file = temp_dir / "2D_CT_model.vtu"
feflow_pv_mesh.save(feflow_mesh_file)

feflow_concentration = Scalar(
    data_name="single_species_P_CONC", data_unit="mg/l", output_unit="mg/l"
)
# The original mesh is clipped to focus on the relevant part of it, where concentration is larger
# than 1e-9 mg/l. The rest of the mesh has concentration values of 0.
mpl.plot(
    feflow_pv_mesh.clip_scalar(
        scalars="single_species_P_CONC", invert=False, value=1.0e-9
    ),
    feflow_concentration,
)

# %%
# 2. Save the point boundary conditions (see: :py:mod:`ogstools.feflowlib.write_point_boundary_conditions`).
write_point_boundary_conditions(temp_dir, feflow_pv_mesh)
# %%
# 3. Setup a prj-file (see: :py:mod:`ogstools.feflowlib.setup_prj_file`) to run a OGS-simulation.
path_prjfile = feflow_mesh_file.with_suffix(".prj")
prjfile = ogs.OGS(PROJECT_FILE=path_prjfile)
species = get_species(feflow_pv_mesh)
CT_model = component_transport(
    saving_path=temp_dir / "sim_2D_CT_model",
    species=species,
    model=prjfile,
    dimension=2,
    fixed_out_times=[48384000],
)
# Include the mesh specific configurations to the template.
model = setup_prj_file(
    bulk_mesh_path=feflow_mesh_file,
    mesh=feflow_pv_mesh,
    material_properties=get_material_properties_of_CT_model(feflow_pv_mesh),
    process="component transport",
    species_list=species,
    model=CT_model,
    initial_time=0,
    end_time=4.8384e07,
    time_stepping=list(zip([10] * 8, [8.64 * 10**i for i in range(8)])),
    max_iter=6,
    rel_tol=1e-14,
)
# The model must be written before it can be run.
model.write_input(path_prjfile)
# Print the prj-file as an example.
ET.dump(ET.parse(path_prjfile))
# %%
# 4. Run the model.
model.run_model(logfile=temp_dir / "out.log")
# %%
# 5. Read the results along a line on the upper edge of the mesh parallel to the x-axis and plot them.
ms = MeshSeries(temp_dir / "sim_2D_CT_model.pvd")
# Read the last timestep:
ogs_sim_res = ms.read(ms.timesteps[-1])
"""
It is also possible to read the file directly with pyvista:
ogs_sim_res = pv.read(
   temp_dir / "sim_2D_CT_model_ts_65_t_48384000.000000.vtu"
)
"""
start_line = (0.038 + 1.0e-8, 0.005, 0)
end_line = (0.045, 0.005, 0)
ogs_line = ogs_sim_res.sample_over_line(start_line, end_line, resolution=100)
feflow_line = feflow_pv_mesh.sample_over_line(
    start_line, end_line, resolution=100
)
plt.rcParams.update({"font.size": 18})
plt.figure()
plt.plot(
    ogs_line.point_data["Distance"] + start_line[0],
    ogs_line.point_data["single_species"],
    linewidth=4,
    color="blue",
    label="ogs",
)
plt.plot(
    feflow_line.point_data["Distance"] + start_line[0],
    feflow_line.point_data["single_species_P_CONC"][:],
    linewidth=4,
    color="black",
    ls=":",
    label="FEFLOW",
)
plt.ylabel("concentration [mg/l]")
plt.xlabel("x [m]")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# %%
# 6. Plot the difference between the FEFLOW and OGS simulation.
plt.figure()
plt.plot(
    ogs_line.point_data["Distance"] + start_line[0],
    feflow_line.point_data["single_species_P_CONC"]
    - ogs_line.point_data["single_species"],
    linewidth=4,
    color="red",
)
plt.ylabel("concentration [mg/l]")
plt.xlabel("x [m]")
plt.title("difference feflow-ogs")
plt.tight_layout()
plt.show()
