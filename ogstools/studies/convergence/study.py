# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path

from ogstools.definitions import ROOT_DIR
from ogstools.workflow import jupytext_to_jupyter


def run_convergence_study(
    output_name: Path,
    mesh_paths: list[Path],
    variable_name: str,
    timevalue: float = 0.0,
    refinement_ratio: float = 2.0,
    reference_solution_path: Path | None = None,
    prepare_only: bool = False,
    show_progress: bool = False,
) -> None:
    """
    Run a convergence study.

    :param output_name:             The output path for the Jupyter notebook.
    :param mesh_paths:              mesh paths of increasing resolutions.
    :param variable_name:           The variable to study for convergence.
    :param timevalue:               The time value for analysis.
    :param refinement_ratio:        The refinement ratio between the meshes.
    :param reference_solution_path: Optional reference solution for comparison.
    :param prepare_only:            If True, don't execute the notebook.
    :param show_progress:           If True, display a progress bar.

    returns: None, but generates the convergence study notebook.
    """
    params = {
        "mesh_paths": [str(mesh_path) for mesh_path in mesh_paths],
        "variable_name": variable_name,
        "timevalue": timevalue,
        "refinement_ratio": refinement_ratio,
        "reference_solution_path": reference_solution_path,
    }
    template_path = Path(
        str(ROOT_DIR) + "/studies/templates/convergence_study.py"
    )
    jupytext_to_jupyter(
        template_path,
        output_name,
        params=params,
        prepare_only=prepare_only,
        show_progress=show_progress,
    )
