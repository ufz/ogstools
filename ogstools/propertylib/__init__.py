# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Define easy-to-access Property classes and PropertyCollection instances."""

from . import mesh_dependent, properties, tensor_math
from .matrix import Matrix
from .property import Property, Scalar
from .vector import Vector

__all__ = [
    "tensor_math",
    "mesh_dependent",
    "properties",
    "Property",
    "Scalar",
    "Vector",
    "Matrix",
]
