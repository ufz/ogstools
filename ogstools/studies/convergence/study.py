from pathlib import Path
from typing import Optional

from ogstools.definitions import ROOT_DIR
from ogstools.workflow import jupytext_to_jupyter


def run_convergence_study(
    output_name: Path,
    mesh_paths: list[Path],
    property_name: str,
    timevalue: float = 0.0,
    refinement_ratio: Optional[float] = None,
    reference_solution_path: Optional[Path] = None,
    prepare_only: bool = False,
    show_progress: bool = False,
) -> None:
    params = {
        "mesh_paths": [str(mesh_path) for mesh_path in mesh_paths],
        "property_name": property_name,
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
