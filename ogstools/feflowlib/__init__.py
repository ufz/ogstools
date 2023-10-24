import logging as log
from importlib.util import find_spec
from sys import stdout

if find_spec("ifm") is None:
    msg = "Could not import ifm. A working FEFLOW Python setup is required!"
    raise ImportError(msg)

from .feflowlib import (
    convert_geometry_mesh,
    convert_properties_mesh,
    update_geometry,
)
from .tools import (
    extract_cell_boundary_conditions,
    get_specific_surface,
    helpFormat,
    setup_prj_file,
    write_point_boundary_conditions,
    write_xml,
)

__all__ = [
    "combine_material_properties",
    "get_specific_surface",
    "helpFormat",
    "convert_geometry_mesh",
    "convert_properties_mesh",
    "update_geometry",
    "extract_cell_boundary_conditions",
    "setup_prj_file",
    "write_point_boundary_conditions",
    "write_xml",
]

# log configuration
log.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=log.INFO,
    stream=stdout,
    datefmt="%d.%m.%Y %H:%M:%S",
)
