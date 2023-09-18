from importlib.util import find_spec

if find_spec("ifm") is None:
    msg = "Could not import ifm. A working FEFLOW Python setup is required!"
    raise ImportError(msg)

from .feflowlib import (
    read_geometry,
    read_properties,
    update_geometry,
)
from .tools import (
    get_specific_surface,
    helpFormat,
    write_cell_boundary_conditions,
    write_point_boundary_conditions,
    write_xml,
)

__all__ = [
    "get_specific_surface",
    "helpFormat",
    "read_geometry",
    "read_properties",
    "update_geometry",
    "write_cell_boundary_conditions",
    "write_point_boundary_conditions",
    "write_xml",
]
