"""Defines the PropertyCollection class.

This class serves as a parent class for the coupled and uncoupled process
classes to group the corresponding properties.
"""

from dataclasses import dataclass
from typing import Literal
from typing import Optional as Opt

from . import defaults
from .property import MatrixProperty, Property, ScalarProperty, VectorProperty


@dataclass(init=False)
class PropertyCollection:
    """Defines a class to group (physical) properties.

    Contains the material_id as a common property for all processes and
    the get_properties method for easy access of all contained properties.
    """

    material_id: ScalarProperty

    def __init__(self):
        """Initialize the PropertyCollection with default attributes."""
        self.material_id = defaults.material_id

    def get_properties(self, dim: Opt[Literal[2, 3]] = None) -> list[Property]:
        """Return all scalar-, vector- or matrix properties."""
        props = []

        for v in self.__dict__.values():
            if not isinstance(v, Property):
                continue
            props += [v]
            if isinstance(v, VectorProperty) and dim in [2, 3]:
                props += [v[i] for i in range(dim)]
            if isinstance(v, MatrixProperty) and dim in [2, 3]:
                props += [v.trace]
                props += [v[i] for i in range(dim * 2)]
        return props

    def find_property(
        self, output_name: str, dim: Opt[Literal[2, 3]] = None
    ) -> Property:
        """Return predefined property with given output_name."""
        for prop in self.get_properties(dim):
            if prop.output_name == output_name:
                return prop
        # if not found by output name, find by data_name
        for prop in self.get_properties(dim):
            if prop.data_name == output_name:
                return prop
        return Property(output_name)
