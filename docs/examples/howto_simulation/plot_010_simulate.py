"""
Run a simulation - from a Project file
======================================

OGSTools is a Python library designed to simplify the process of running
simulations with the OpenGeoSys (OGS) framework. The `Project` class is a core
component of OGSTools, providing a convenient interface for creating and
altering OGS6 input files up and executing OGS simulations. This allows you to
automate OGS-workflows in Python via Jupyter or just plain Python scripts.
The development of this functionality was first started in
[ogs6py](https://github.com/joergbuchwald/ogs6py/) and is continued in OGSTools.
Here you'll find the detailed API: :py:obj:`ogstools.ogs6py.project.Project`.

Features:
- alternate existing files (e.g., for parameter sweeps)
- create new input files from scratch
- execute project files
- tailored alteration of input files e.g. for mesh replacements or restarts
- display and export parameter settings

In this guide, we will walk you through the process of using the `Project` class
to run a simple simulation from an existing model setup.
Assuming you have prepared a model with your mesh and a project file you can
use the following setup, to run it from python.
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import ogstools as ot
from ogstools.definitions import EXAMPLES_DIR

results_dir = Path(mkdtemp())
prj_path_in = EXAMPLES_DIR / "prj" / "simple_mechanics.prj"
prj_path_out = results_dir / "simple_mechanics_modified.prj"
prj = ot.Project(
    input_file=prj_path_in, output_file=prj_path_out, output_dir=results_dir
)
prj.write_input()
# as the current working directory of this notebook is not the same as the
# directory of the prj-file and the meshes, we have to tell OGS via the flag
# "-m", that the meshes are in the same directory as the input_file
# "-o" sets the output_directory
prj.run_model(args=f"-m {prj_path_in.parent} -o {results_dir}")

# %% [markdown]
# Manipulating the Project
# ========================
# By using a prj-file as a template and modifying it in python we have an easy
# way to parametrize simulations. Below are some methods, which change
# different parts of the model definition. For more detailed information have
# a look into the API.

# %%
prj.replace_parameter_value(name="E", value=1e9)
# You can achieve the same via the `replace_text` method:
prj.replace_text(1e9, xpath="./parameters/parameter[name='E']/value")
# Let's also replace the output prefix of the result
prj.replace_text("E=1e9", xpath="./time_loop/output/prefix")
# The density of a phase can also be changed
prj.replace_phase_property_value(
    mediumid=0, phase="Solid", name="density", value="42"
)

# %% [markdown]
# After modifying the Project you can execute the model in the same way as
# before. You have to run the `write_input` method beforehand again. The changes
# will not be reflected in the simulation otherwise. This step will be
# automated in the future.

# %% [markdown]
# Background execution
# ====================
# To execute a simulation in the background, so that your python environment is
# not frozen until the simulation finishes, you can set `background` to True.

# %%
prj.write_input()
prj.run_model(background=True, args=f"-m {prj_path_in.parent} -o {results_dir}")
print(prj.status)

# %% [markdown]
# If you want to abort the simulation for any reason, use the following command:

# %%
prj.terminate_run()
print(prj.status)

# %% [markdown]
# Creating a Project from scratch
# ===============================
# You can also create a Project without a prj-file. Have a look at this example
# to see how: :ref:`sphx_glr_auto_examples_howto_prjfile_plot_creation.py`
