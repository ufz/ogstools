import argparse
import tempfile
from pathlib import Path

import jupytext
import papermill as pm


def execute_convergence_study(
    output_path: str, mesh_paths: list[str], property_name: str, timestep: int
) -> None:
    params = {
        "mesh_paths": mesh_paths,
        "property_name": property_name,
        "timestep": timestep,
    }
    parent = Path(__file__).resolve().parent
    template_path = str(parent) + "/convergence_study_template.md"
    nb = jupytext.read(template_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as temp:
        jupytext.write(nb, temp.name, fmt="py:percent")
    pm.execute_notebook(
        input_path=temp.name,
        output_path=output_path,
        parameters=params,
    )
    Path(temp.name).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence Study")
    parser.add_argument("output", help="Path to the output notebook")
    parser.add_argument("--mesh_paths", nargs="+", help="List of mesh paths")
    parser.add_argument("--property_name", help="Name of the property")
    parser.add_argument("--timestep", help="Timestep to read")

    args = parser.parse_args()

    execute_convergence_study(
        args.output, args.mesh_paths, args.property_name, args.timestep
    )
