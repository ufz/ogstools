# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import math
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import gmsh
import yaml  # type: ignore[import]
from simpleeval import simple_eval


def _load_geometry_from_yaml(path: Path) -> dict[str, Any]:
    """Load geometry definition from YAML file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_eval_context(
    params: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare context for simple_eval with math constants and functions."""
    names = {
        **params,
        **{k: v for k, v in vars(math).items() if not k.startswith("_")},
    }
    functions = {k: v for k, v in vars(math).items() if callable(v)}
    return names, functions


def _evaluate_expr(
    expr: Any, params: dict[str, Any]
) -> int | float | str | list:
    """
    Evaluates an expression (str, list, int, float).
    Allows mathematical functions and variables.
    """
    names, functions = _make_eval_context(params)

    if isinstance(expr, (int | float)):
        return expr

    if isinstance(expr, str):
        try:
            return simple_eval(expr, names=names, functions=functions)
        except Exception as e:
            msg = f"Error evaluating expression '{expr}': {e}"
            raise ValueError(msg) from e

    if isinstance(expr, list):
        results = []
        for item in expr:
            if isinstance(item, str):
                try:
                    results.append(
                        simple_eval(item, names=names, functions=functions)
                    )
                except Exception as e:
                    msg = f"Error evaluating expression '{item}': {e}"
                    raise ValueError(msg) from e
            else:
                results.append(item)
        return results

    msg = f"Unsupported type for expression: {type(expr)}. Expected str, list, int, or float."
    raise TypeError(msg)


def _evaluate_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate all string expressions in the params dict using _evaluate_expr.
    Updates the dict in-place with numeric results.
    Raises ValueError with a consistent message if evaluation fails.
    """
    for key, value in list(params.items()):
        if isinstance(value, str):
            try:
                params[key] = _evaluate_expr(value, params)
            except Exception as e:
                msg = f"Failed to evaluate parameter '{key}': {e}"
                raise ValueError(msg) from e
    return params


def _validate_block(geometry: dict[str, Any], block_name: str) -> None:
    if block_name not in geometry:
        msg = f"Missing top-level block: '{block_name}'"
        raise ValueError(msg)

    block = geometry[block_name]
    if not isinstance(block, dict):
        msg = f"Block '{block_name}' must be a mapping, got {type(block).__name__}"
        raise ValueError(msg)

    for name, definition in block.items():
        if not isinstance(definition, dict):
            msg = f"{block_name[:-1].capitalize()} '{name}' must be a mapping, got {type(definition).__name__}"
            raise ValueError(msg)


def _validate_geometry_schema(geometry: dict[str, Any]) -> None:
    """Ensure required blocks exist and have the correct types."""
    required_blocks = ["points", "lines", "surfaces", "groups"]
    for key in required_blocks:
        _validate_block(geometry, key)


def meshes_from_yaml(
    geometry_file: Path, output_dir: Path | None = None
) -> Path:
    """
    Generate a 2D mesh from a YAML geometry file using gmsh.
    """
    output_dir = output_dir or Path(mkdtemp())

    geometry = _load_geometry_from_yaml(geometry_file)
    _validate_geometry_schema(geometry)

    params = _evaluate_parameters(geometry.get("parameters") or {})

    gmsh.initialize()
    gmsh.model.add("simple_mesh")

    # --- Points ---
    gmsh_points: dict[str, int] = {}
    for name, val in geometry["points"].items():
        if "coords" not in val:
            msg = f"Incomplete point definition for {name}"
            raise ValueError(msg)
        coords = val["coords"]

        # make sure coords is a list or a tuple.
        if not isinstance(coords, (list | tuple)):
            msg = f"Point {name}: 'coords' must be a list of [x, y], got {type(coords).__name__}"
            raise ValueError(msg)
        if len(coords) != 2:
            msg = f"Point {name}: 'coords' must have exactly 2 entries [x, y], but got {len(coords)} values: {coords}"
            raise ValueError(msg)

        x, y = coords
        x = _evaluate_expr(x, params)
        y = _evaluate_expr(y, params)
        char_length = _evaluate_expr(val["char_length"], params)
        gmsh_points[name] = gmsh.model.geo.addPoint(x, y, 0.0, char_length)

    # --- Lines & arcs ---
    gmsh_lines: dict[str, int] = {}
    for name, val in geometry["lines"].items():
        start = gmsh_points[val["start"]]
        end = gmsh_points[val["end"]]
        if "center" in val:
            center = gmsh_points[val["center"]]
            gmsh_lines[name] = gmsh.model.geo.addCircleArc(start, center, end)
        else:
            gmsh_lines[name] = gmsh.model.geo.addLine(start, end)

    # --- Surfaces ---
    gmsh_surfaces: dict[str, int] = {}
    for sname, sdef in geometry["surfaces"].items():
        if "loop" in sdef:
            loop = [
                (
                    -gmsh_lines[lref[1:]]
                    if lref.startswith("-")
                    else gmsh_lines[lref]
                )
                for lref in sdef["loop"]
            ]
            loops = [gmsh.model.geo.addCurveLoop(loop)]
        elif "loops" in sdef:
            loops = []
            for subloop in sdef["loops"]:
                loop = [
                    (
                        -gmsh_lines[lref[1:]]
                        if lref.startswith("-")
                        else gmsh_lines[lref]
                    )
                    for lref in subloop
                ]
                loops.append(gmsh.model.geo.addCurveLoop(loop))
        else:
            msg = f"Surface {sname} has no 'loop' or 'loops'."
            raise ValueError(msg)
        gmsh_surfaces[sname] = gmsh.model.geo.addPlaneSurface(loops)

    gmsh.model.geo.synchronize()

    # --- Groups ---
    for gname, gdef in geometry["groups"].items():
        if gdef["dim"] == 2:
            entity_ids = [gmsh_surfaces[e] for e in gdef["entities"]]
        elif gdef["dim"] == 1:
            entity_ids = [gmsh_lines[e] for e in gdef["entities"]]
        elif gdef["dim"] == 0:
            entity_ids = [gmsh_points[e] for e in gdef["entities"]]
        else:
            msg = f"Unsupported group dimension: {gdef['dim']}"
            raise NotImplementedError(msg)
        tag = gmsh.model.addPhysicalGroup(gdef["dim"], entity_ids)
        gmsh.model.setPhysicalName(gdef["dim"], tag, gname)

    # --- Mesh output ---
    gmsh.model.mesh.generate(2)
    logs = gmsh.logger.get()
    for msg in logs:
        print("GMSH:", msg)
    output_file = Path(output_dir) / "mesh.msh"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_file))
    gmsh.finalize()

    return output_file


__all__ = ["meshes_from_yaml"]
