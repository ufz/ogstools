# Author: Florian Zill (Helmholtz Centre for Environmental Research GmbH - UFZ)
"""Define easy-to-access Property classes and PropertyCollection instances."""

from . import presets
from .property import Matrix, Property, Scalar, Vector

__all__ = [
    "presets",
    "Property",
    "Scalar",
    "Vector",
    "Matrix",
]
