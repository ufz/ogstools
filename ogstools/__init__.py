# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

""" """

from importlib import metadata

__version__ = metadata.version(__package__)
__authors__ = metadata.metadata(__package__)["Author-email"]

del metadata  # optional, avoids polluting the results of dir(__package__)

try:
    from . import plot, variables
    from .meshlib import Mesh, MeshSeries  # noqa: F401
    from .ogs6py import Project  # noqa: F401
except ImportError as e:
    print(f"Warning: Failed to import the following libraries: {e.name}")


__all__ = [
    "plot",
    # "Mesh",
    # "MeshSeries",
    "variables",
]
