from importlib.util import find_spec

if find_spec("ifm") is None:
    msg = "Could not import ifm. A working Feflow Python setup is required!"
    raise ImportError(msg)

from .fe2vtu import get_matids_from_selections, get_pt_cell_data, get_pts_cells
from .tools import (
    get_specific_surface,
    write_xml,
)

__all__ = [
    "get_matids_from_selections",
    "get_pt_cell_data",
    "get_pts_cells",
    "get_specific_surface",
    "write_xml",
]
