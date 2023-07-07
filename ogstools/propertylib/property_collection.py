"""Defines the PropertyCollection class.

This class serves as a parent class for the coupled and uncoupled process
classes to group the corresponding properties.
"""

from dataclasses import dataclass
from typing import Union

from .property import MatrixProperty, ScalarProperty, VectorProperty


@dataclass(init=False)
class PropertyCollection:
    """Defines a class to group (physical) properties.

    Contains the material_id as a common property for all processes and
    the get_properties method for easy access of all contained properties.
    """

    material_id: ScalarProperty

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        self.material_id = ScalarProperty("MaterialIDs")

    def get_properties(
        self,
    ) -> list[Union[ScalarProperty, VectorProperty, MatrixProperty]]:
        """Return all scalar-, vector- or matrix properties."""
        return [
            v
            for v in self.__dict__.values()
            if isinstance(v, (ScalarProperty, VectorProperty, MatrixProperty))
        ]
