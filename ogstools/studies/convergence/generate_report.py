# make this runnable via python and CLI with argparse

from pathlib import Path

import jupytext
import papermill as pm

from ogstools.studies.convergence.examples import mesh_paths

parent = Path(__file__).resolve().parent

nb = jupytext.read("convergence_study.py")
# TODO: make temporary
jupytext.write(nb, "convergence_study.ipynb", fmt="py:percent")

property_name = "pressure"
params = {"mesh_paths": mesh_paths, "property_name": property_name}
pm.execute_notebook(
    input_path=str(parent) + "/convergence_study.ipynb",
    output_path="convergence_study_" + property_name + ".ipynb",
    parameters=params,
)
