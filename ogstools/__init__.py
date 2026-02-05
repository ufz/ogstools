# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from contextlib import suppress
from importlib import metadata

# Top API
from . import gmsh_tools, logparser, mesh, plot, variables
from ._find_ogs import cli, status  # noqa: F401
from .core.execution import Execution
from .core.interactive_simulation_controller import (
    OGSInteractiveController,  # noqa: F401
)
from .core.model import Model
from .core.native_simulation_controller import OGSNativeController  # noqa: F401
from .core.result import Result  # noqa: F401
from .core.simulation import Simulation
from .core.storage import StorageBase  # noqa: F401
from .logparser.log import Log
from .materiallib.core.material_manager import MaterialManager
from .materiallib.core.media import MediaSet
from .meshes._meshes import Meshes
from .meshseries._meshseries import MeshSeries
from .ogs6py.project import Project

with suppress(ImportError):
    from .feflowlib import FeflowModel  # noqa: F401


__version__ = metadata.version(__package__)
__authors__ = metadata.metadata(__package__)["Author-email"]

del metadata  # optional, avoids polluting the results of dir(__package__)

""".. noindex::"""

__all__ = [
    "Execution",
    "Log",
    "MaterialManager",
    "MediaSet",
    "MeshSeries",
    "Meshes",
    "Model",
    "Project",
    "Simulation",
    "gmsh_tools",
    "logparser",
    "mesh",
    "plot",
    "variables",
]
