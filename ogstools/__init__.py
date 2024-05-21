# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""
"""
from importlib import metadata

from .propertylib import properties

__version__ = metadata.version(__package__)
__authors__ = metadata.metadata(__package__)["Author-email"]

del metadata  # optional, avoids polluting the results of dir(__package__)


__all__ = [
    "properties",
]
