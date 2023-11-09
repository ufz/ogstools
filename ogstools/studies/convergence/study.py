import argparse
from pathlib import Path
from typing import Optional

from ogstools.definitions import ROOT_DIR
from ogstools.workflow import jupytext_to_jupyter


def run_convergence_study(
    output_name: Path,
    mesh_paths: list[Path],
    topology_path: Path,
    property_name: str,
    timestep: int = 0,
    refinement_ratio: Optional[float] = None,
    reference_solution_path: Optional[Path] = None,
    prepare_only: bool = False,
    progress_bar: bool = False,
) -> None:
    params = {
        "mesh_paths": [str(mesh_path) for mesh_path in mesh_paths],
        "topology_path": str(topology_path),
        "property_name": property_name,
        "timestep": timestep,
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
        progress_bar=progress_bar,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence Study")
    parser.add_argument("output", help="Path to the output notebook")
    parser.add_argument("--mesh_paths", nargs="+", help="List of mesh paths")
    parser.add_argument("--topology_path", help="Path to topology mesh")
    parser.add_argument("--property_name", help="Name of the property")
    parser.add_argument("--timestep", help="Timestep to read")

    args = parser.parse_args()

    run_convergence_study(
        args.output,
        args.mesh_paths,
        args.topology_path,
        args.property_name,
        args.timestep,
    )
