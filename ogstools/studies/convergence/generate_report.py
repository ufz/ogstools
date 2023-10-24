# make this runnable via python and CLI with argparse

from pathlib import Path

import papermill as pm

from ogstools.studies.convergence.examples import mesh_paths

parent = Path(__file__).resolve().parent

property_name = "pressure"
params = {"mesh_paths": mesh_paths, "property_name": property_name}
pm.execute_notebook(
    str(parent) + "/template.ipynb", "report.ipynb", parameters=params
)
