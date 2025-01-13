"""
Feflowlib: How to modify the project-file after converting a FEFLOW model.
==========================================================================

.. sectionauthor:: Julian Heinze (Helmholtz Centre for Environmental Research GmbH - UFZ)

This example shows how to convert a FEFLOW model and how to modify the corresponding OGS project file and boundary meshes after conversion.
"""

# %%
# 0. Necessary imports
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pyvista as pv

import ogstools as ot
from ogstools.examples import feflow_model_2D_CT_t_560
from ogstools.feflowlib import FeflowModel

# %%
# 1. Load a FEFLOW model (.fem) as a FeflowModel object to further work it.
# During the initialisation, the FEFLOW file is converted.
temp_dir = Path(tempfile.mkdtemp("converted_models"))
feflow_model = FeflowModel(feflow_model_2D_CT_t_560, temp_dir / "CT_model")

# %%
# 2. Get the mesh and run the FEFLOW model in OGS.
mesh = feflow_model.mesh
feflow_model.setup_prj(
    end_time=1e11, time_stepping=[(1, 1e6), (1, 1e8), (1, 1e10)]
)
feflow_model.run()
# %%
# 3.Plot the results.
ms = ot.MeshSeries(temp_dir / "CT_model.pvd")
ogs_sim_res = ms.mesh(ms.timesteps[-1])
ot.plot.contourf(ogs_sim_res, "single_species")
# %%
# 4. Replace the scalar pore diffusion constant by a tensor to introducec anisotropy.
# How to manipulate a prj file also is explained in this example: :ref:sphx_glr_auto_examples_howto_prjfile_plot_manipulation.py.
project = feflow_model.project
# tensor = "3e-10"
tensor = """
        1e-9 1e-12
        """
project.replace_phase_property_value(
    mediumid=0, component="single_species", name="pore_diffusion", value=tensor
)
project.write_input()
model_prjfile = ET.parse(temp_dir / "CT_model.prj")
ET.dump(model_prjfile)
# %%
# 5. Remove some points of the boundary mesh.
boundary_conditions = feflow_model.boundary_conditions
bounds = [0.037, 0.039, 0.003, 0.006, 0, 0]
new_bc = boundary_conditions[
    str(temp_dir / "single_species_P_BC_MASS.vtu")
].clip_box(bounds, invert=False)
pv.save_meshio(str(temp_dir / "single_species_P_BC_MASS.vtu"), new_bc)

# %%
# 5. Run the model.
project.run_model()
ms = ot.MeshSeries(temp_dir / "CT_model.pvd")
ogs_sim_res = ms.mesh(ms.timesteps[-1])
ot.plot.contourf(ogs_sim_res, "single_species")
