# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from contextlib import suppress
from importlib import metadata

from . import plot, variables
from .meshlib import Mesh, MeshSeries, meshes_from_gmsh  # noqa: F401
from .ogs6py import Project  # noqa: F401

with suppress(ImportError):
    from .feflowlib import FeflowModel  # noqa: F401

__version__ = metadata.version(__package__)
__authors__ = metadata.metadata(__package__)["Author-email"]

del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "plot",
    "variables",
]
