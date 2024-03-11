from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from pint.facets.plain import PlainQuantity

from ogstools.propertylib.property import Property, Scalar

from .unit_registry import u_reg

ValType = Union[PlainQuantity, np.ndarray]


def vector_norm(vals: ValType) -> ValType:
    ":returns: The norm of the vector."
    if isinstance(vals, PlainQuantity):
        unit = vals.units
        vals = vals.magnitude
    else:
        unit = None
    result = np.linalg.norm(vals, axis=-1)
    return result if unit is None else u_reg.Quantity(result, unit)


@dataclass
class Vector(Property):
    """Represent a vector property of a dataset.

    Vector properties should contain either 2 (2D) or 3 (3D) components.
    Vector components can be accesses with brackets e.g. displacement[0]
    """

    def __getitem__(self, index: Union[int, Literal["x", "y", "z"]]) -> Scalar:
        """
        Get a scalar property as a specific component of the vector property.

        :param index: The index of the component.

        :returns: A scalar property as a vector component.
        """
        int_index = index if isinstance(index, int) else "xyz".index(index)
        return Scalar.from_property(
            self,
            output_name=self.output_name + f"_{index}",
            func=lambda x: self.func(x)[..., int_index],
            bilinear_cmap=True,
        )

    @property
    def magnitude(self) -> Scalar:
        ":returns: A scalar property as the magnitude of the vector."
        return Scalar.from_property(
            self,
            output_name=self.output_name + "_magnitude",
            func=lambda x: vector_norm(self.func(x)),
        )


@dataclass
class VectorList(Property):
    """Represent a list of vector properties of a dataset."""

    def __getitem__(self, index: int) -> Vector:
        ":returns: A vector property as a component of the vectorlist property."
        return Vector.from_property(
            self,
            output_name=self.output_name + f"_{index}",
            func=lambda x: np.take(self.func(x), index, axis=-1),
        )
