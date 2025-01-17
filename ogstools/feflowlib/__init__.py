# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

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
from ._prj_tools import setup_prj_file
from ._templates import (
    component_transport,
    hydro_thermal,
    liquid_flow,
    steady_state_diffusion,
)
from ._tools import (
    assign_bulk_ids,
    extract_cell_boundary_conditions,
    extract_point_boundary_conditions,
    get_material_properties_of_CT_model,
    get_material_properties_of_HT_model,
    get_species,
    get_specific_surface,
    helpFormat,
    remove_bulk_ids,
    write_point_boundary_conditions,
)
from .feflow_model import FeflowModel

__all__ = [
    "assign_bulk_ids",
    "component_transport",
    "convert_geometry_mesh",
    "convert_properties_mesh",
    "extract_cell_boundary_conditions",
    "extract_point_boundary_conditions",
    "FeflowModel",
    "get_material_properties_of_CT_model",
    "get_material_properties_of_HT_model",
    "get_species",
    "get_specific_surface",
    "helpFormat",
    "hydro_thermal",
    "liquid_flow",
    "points_and_cells",
    "remove_bulk_ids",
    "setup_prj_file",
    "steady_state_diffusion",
    "update_geometry",
    "write_point_boundary_conditions",
]

# log configuration
log.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=log.WARN,
    stream=stdout,
    datefmt="%d.%m.%Y %H:%M:%S",
)
