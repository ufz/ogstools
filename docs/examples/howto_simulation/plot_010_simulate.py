"""
Run a simulation
================

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
and then run a simple simulation from an existing model setup.
Assuming you have prepared a model with your mesh and a project file you can
use the following setup, to run it from python.

**Choosing the OGS binary**

Execution details (which OGS binary to use, MPI settings, logging, …) are
controlled via :py:class:`ogstools.core.execution.Execution`.  Pre-built OGS
binaries and container images for each release are available at
`<https://www.opengeosys.org/6.5.7/releases/>`_.

To apply site-wide defaults on a shared system (e.g. HPC cluster), set the
``OGS_EXECUTION_DEFAULTS`` environment variable to a YAML file that overrides
only the keys you need::

    export OGS_EXECUTION_DEFAULTS=/path/to/my_defaults.yaml

See :py:class:`ogstools.core.execution.Execution` for the full list of options.
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

model = ot.Model(prj, meshes=EXAMPLES_DIR / "prj")
sim = model.run()
# Optionally save the simulation data
sim.save()
print(sim)


# %% [markdown]
# Manipulating the Project
# ========================
# By using a prj-file as a template and modifying it in python we have an easy
# way to parametrize simulations. Below are some methods, which change
# different parts of the model definition. For more detailed information have
# a look into the API.
# The subsequent code would work but for clarity we recommend saving 2 different states of the prj object into 2 different files.

# Either tell that you are going to change prj object (prj.copy) OR do prj.save() after you have changed but before you run the simulation.
prj2 = prj.copy()


# %%
prj2.replace_parameter_value(name="E", value=1e9)
# You can achieve the same via the `replace_text` method:
prj2.replace_text(1e9, xpath="./parameters/parameter[name='E']/value")
# Let's also replace the output prefix of the result
prj2.replace_text("E=1e9", xpath="./time_loop/output/prefix")
# The density of a phase can also be changed
prj2.replace_phase_property_value(
    mediumid=0, phase="Solid", name="density", value="42"
)

# %% [markdown]
# After modifying the Project you can execute the model in the same way as
# before. You have to prj.save(new_name) here, or beforehand by prj2.copy.

model2 = ot.Model(prj2, meshes=model.meshes)
sim = model2.run()
print(sim)

# %%
# Alternatively, for more control
simc = model2.controller()  # this call is not blocking
simc.terminate()  # do something while the simulation is running
simc.run()  # this call is blocking, it waits for the simulation to finish


# %% [markdown]
# Logfile monitoring
# ====================
# If you want to plot the progress of your simulation in real-time, you can use the
# 'plot_log' method from the returned `Monitor` object. The model run has to be
# executed in the background for this to work.

# %%
prj.write_input()
prj.run_model(background=True, args=f"-m {prj_path_in.parent} -o {results_dir}")
prj.plot_log(
    log_data=[["step_start_time", "step_size"], ["iteration_number", "dx_x"]]
)

# %% [markdown]
# .. image:: /examples/howto_simulation/bokeh_logs.png
#    :alt: Bokeh log plot

# %% [markdown]
# Creating a Project from scratch
# ===============================
# You can also create a Project without a prj-file. Have a look at this example
# to see how: :ref:`sphx_glr_auto_examples_howto_prjfile_plot_creation.py`
