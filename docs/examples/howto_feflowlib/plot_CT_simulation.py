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
import pyvista as pv
from ogs6py import ogs

import ogstools.meshplotlib as mpl
from ogstools.examples import feflow_model_2D_CT_560
from ogstools.feflowlib import (
    component_transport,
    convert_properties_mesh,
    extract_point_boundary_conditions,
    get_material_properties_of_CT_model,
    get_species,
    setup_prj_file,
    write_point_boundary_conditions,
)
from ogstools.propertylib import Scalar

# %%
# 1. Load a FEFLOW model (.fem) as a FEFLOW document, convert and save it.
feflow_model = ifm.loadDocument(str(feflow_model_2D_CT_560))
feflow_pv_mesh = convert_properties_mesh(feflow_model)

path_writing = Path(tempfile.mkdtemp("feflow_test_simulation"))
path_mesh = path_writing / "2D_CT_model.vtu"
feflow_pv_mesh.save(path_mesh)

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
# 2. Extract the point boundary conditions and save them.

point_BC_dict = extract_point_boundary_conditions(path_writing, feflow_pv_mesh)
write_point_boundary_conditions(path_writing, feflow_pv_mesh)
# %%
# 3. Setup a prj-file to run a OGS-simulation
path_prjfile = str(path_mesh.with_suffix(".prj"))
prjfile = ogs.OGS(PROJECT_FILE=str(path_prjfile))
species = get_species(feflow_pv_mesh)
CT_model = component_transport(
    path_writing / "sim_2D_CT_model",
    species,
    prjfile,
    dimension2D=True,
)
# Include the mesh specific configurations to the template.
model = setup_prj_file(
    path_mesh,
    feflow_pv_mesh,
    get_material_properties_of_CT_model(feflow_pv_mesh),
    "component transport",
    species_list=species,
    model=CT_model,
)
# The model must be written before it can be run.
model.write_input(str(path_prjfile))
# Print the prj-file as an example.
model_prjfile = ET.parse(str(path_prjfile))
ET.dump(model_prjfile)
# %%
# 4. Run the model.
model.run_model(logfile=str(path_writing / "out.log"))
# %%
# 5. Read the results along a line on the upper edge of the mesh parallel to the x-axis and plot them.
ogs_simulation = pv.read(
    str(path_writing / "sim_2D_CT_model_ts_71_t_48384000.000000.vtu")
)
start = (0.038 + 1.0e-8, 0.01, 0)
end = (1, 0.01, 0)
ogs_line = ogs_simulation.sample_over_line(start, end, resolution=10000)
feflow_line = feflow_pv_mesh.sample_over_line(start, end, resolution=10000)
plt.rcParams.update({"font.size": 18})
plt.figure()
plt.plot(
    ogs_line.point_data["Distance"] + 0.038 + 1.0e-8,
    ogs_line.point_data["single_species"],
    linewidth=4,
    color="blue",
    label="ogs",
)
plt.plot(
    feflow_line.point_data["Distance"] + 0.038 + 1.0e-8,
    feflow_line.point_data["single_species_P_CONC"][:],
    linewidth=4,
    color="black",
    ls=":",
    label="FEFLOW",
)
plt.ylabel("concentration [mg/l]")
plt.xlabel("x [m]")
plt.xlim(0.0375, 0.045)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# %%
# 6. Plot the difference between the FEFLOW and OGS simulation.
plt.figure()
plt.plot(
    ogs_line.point_data["Distance"] + 0.038 + 1.0e-8,
    feflow_line.point_data["single_species_P_CONC"]
    - ogs_line.point_data["single_species"],
    linewidth=4,
    color="red",
)
plt.ylabel("concentration [mg/l]")
plt.xlabel("x [m]")
plt.title("difference feflow-ogs")
plt.xlim(0.0375, 0.045)
plt.tight_layout()
plt.show()
