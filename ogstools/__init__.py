# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from contextlib import suppress
from importlib import metadata

from . import logparser, plot, variables
from ._find_ogs import cli, status
from .materiallib.core.material_manager import MaterialManager  # noqa: F401
from .materiallib.core.media import MediaSet  # noqa: F401
from .meshlib import (
    Mesh,  # noqa: F401
    Meshes,  # noqa: F401
    MeshSeries,  # noqa: F401
    meshes_from_gmsh,
)
from .ogs6py import Project  # noqa: F401

with suppress(ImportError):
    from .feflowlib import FeflowModel  # noqa: F401


__version__ = metadata.version(__package__)
__authors__ = metadata.metadata(__package__)["Author-email"]

del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "cli",
    "logparser",
    "plot",
    "status",
    "variables",
    "meshes_from_gmsh",
]
