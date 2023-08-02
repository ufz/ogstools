from importlib.util import find_spec

if find_spec("ifm") is None:
    msg = "Could not import ifm. A working Feflow Python setup is required!"
    raise ImportError(msg)

from .fe2vtu import (
    get_geo_mesh,
    get_property_mesh,
    update_geo_mesh,
)
from .tools import get_specific_surface, helpFormat, write_pt_bc, write_xml

__all__ = [
    "get_geo_mesh",
    "get_specific_surface",
    "get_property_mesh",
    "helpFormat",
    "update_geo_mesh",
    "write_xml",
    "write_pt_bc",
]
