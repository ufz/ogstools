import logging as log
from importlib.util import find_spec
from sys import stdout

if find_spec("ifm") is None:
    msg = "Could not import ifm. A working FEFLOW Python setup is required!"
    raise ImportError(msg)

from ._feflowlib import (
    convert_geometry_mesh,
    convert_properties_mesh,
    points_and_cells,
    update_geometry,
)
from .templates import liquid_flow, steady_state_diffusion
from .tools import (
    combine_material_properties,
    deactivate_cells,
    extract_cell_boundary_conditions,
    extract_point_boundary_conditions,
    get_specific_surface,
    helpFormat,
    setup_prj_file,
    write_mesh_of_combined_properties,
    write_point_boundary_conditions,
)

__all__ = [
    "combine_material_properties",
    "convert_geometry_mesh",
    "convert_properties_mesh",
    "deactivate_cells",
    "extract_cell_boundary_conditions",
    "extract_point_boundary_conditions",
    "get_specific_surface",
    "helpFormat",
    "liquid_flow",
    "points_and_cells",
    "setup_prj_file",
    "steady_state_diffusion",
    "update_geometry",
    "write_mesh_of_combined_properties",
    "write_point_boundary_conditions",
]

# log configuration
log.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=log.INFO,
    stream=stdout,
    datefmt="%d.%m.%Y %H:%M:%S",
)
