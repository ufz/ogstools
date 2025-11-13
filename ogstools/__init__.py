# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from contextlib import suppress
from importlib import metadata

from . import gmsh_tools, logparser, mesh, plot, variables
from ._find_ogs import cli, status
from .materiallib.core.material_manager import MaterialManager
from .materiallib.core.media import MediaSet
from .meshes import Meshes
from .meshseries import MeshSeries
from .ogs6py import Project

with suppress(ImportError):
    from .feflowlib import FeflowModel  # noqa: F401


__version__ = metadata.version(__package__)
__authors__ = metadata.metadata(__package__)["Author-email"]

del metadata  # optional, avoids polluting the results of dir(__package__)

""".. noindex::"""

__all__ = [
    "MaterialManager",
    "MediaSet",
    "MeshSeries",
    "Meshes",
    "Project",
    "cli",
    "gmsh_tools",
    "logparser",
    "mesh",
    "plot",
    "status",
    "variables",
]
