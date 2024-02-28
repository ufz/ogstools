# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Define easy-to-access Property classes and PropertyCollection instances."""

from . import mesh_dependent, presets, tensor_math
from .matrix import Matrix
from .property import Property, Scalar
from .vector import Vector

__all__ = [
    "tensor_math",
    "mesh_dependent",
    "presets",
    "Property",
    "Scalar",
    "Vector",
    "Matrix",
]
