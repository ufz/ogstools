"""Defines the Scalar, Vector and Matrix Property classes.

They serve as classes to handle common physical properties in a systematic
way (e.g. temperature, pressure, displacement, …). Unit conversion is handled
via pint.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable, Union

import numpy as np
from pint import UnitRegistry
from pint.facets.plain import PlainQuantity

from .utils import identity, sym_tensor_to_mat
from .vector2scalar import trace

u_reg: UnitRegistry = UnitRegistry(
    preprocessors=[lambda s: s.replace("%", "percent")]
)
u_reg.default_format = "~.12g"
u_reg.setup_matplotlib(True)


# TODO: rename to BaseProperty?? / GenericProperty
@dataclass
class Property:
    """Represent a property of a dataset."""

    data_name: str
    """The name of the property data in the dataset."""
    data_unit: str = ""
    """The unit of the property data in the dataset."""
    output_unit: str = ""
    """The output unit of the property."""
    output_name: str = ""
    """The output name of the property."""
    mask: str = ""
    """The name of the mask data in the dataset."""
    func: Union[
        Callable[
            [Union[float, np.ndarray, PlainQuantity]],
            Union[float, np.ndarray, PlainQuantity],
        ],
        Callable[[Any], Any],
    ] = identity
    """The function to be applied on the data."""
    bilinear_cmap: bool = False
    """Should this property be displayed with a bilinear cmap?"""
    categoric: bool = False
    """Does this property only have categoric values?"""

    def __post_init__(self):
        if not self.output_name:
            self.output_name = self.data_name

    @property
    def type_name(self):
        return type(self).__name__

    def replace(self, **changes):
        """
        Create a new Property object with modified attributes.

        Be aware that there is no type check safety here. So make sure, the new
        attributes and values are correct.

        :param changes: Attributes to be changed.

        :returns: A copy of the Property with changed attributes.
        """
        return replace(self, **changes)

    def __call__(self, vals: np.ndarray) -> PlainQuantity:
        """
        Return transformed values with units.

        Apply property function and convert from data_unit to output_unit

        :param vals: The input values.

        :returns: The values with units.
        """
        Q_, _du, _ou = u_reg.Quantity, self.data_unit, self.output_unit
        if Q_(0, _du).dimensionality == Q_(0, _ou).dimensionality:
            return Q_(Q_(self.func(np.asarray(vals)), _du), _ou)
        return Q_(self.func(Q_(vals, _du)), _ou)

    def strip_units(self, vals: np.ndarray) -> np.ndarray:
        """
        Return transformed values without units.

        Apply property function, convert from data_unit to output_unit and
        strip the unit.

        :param vals: The input values.

        :returns: The values without units.
        """
        return self(vals).magnitude

    def get_output_unit(self) -> str:
        """
        Get the output unit.

        returns: The output unit.
        """
        return "%" if self.output_unit == "percent" else self.output_unit

    def is_mask(self) -> bool:
        """
        Check if the property is a mask.

        :returns: True if the property is a mask, False otherwise.
        """
        return self.data_name == self.mask

    def get_mask(self):
        """
        :returns: A property representing this properties mask.
        """
        return Property(data_name=self.mask, mask=self.mask, categoric=True)

    @property
    def magnitude(self) -> "Property":
        return self


@dataclass
class Scalar(Property):
    "Represent a scalar property of a dataset."


@dataclass
class Vector(Property):
    """Represent a vector property of a dataset.

    Vector properties should contain either 2 (2D) or 3 (3D) components.
    Vector components can be accesses with brackets e.g. displacement[0]
    """

    def __getitem__(self, index: int) -> Scalar:
        """
        Get a scalar property as a specific component of the vector property.

        :param index: The index of the component.

        :returns: A scalar property as a vector component.
        """
        suffix = {False: index, True: ["x", "y", "z"][index]}
        return Scalar(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + f"_{suffix[0 <= index <= 2]}",
            mask=self.mask,
            func=lambda x: np.array(x)[..., index],
            bilinear_cmap=True,
        )

    @property
    def magnitude(self) -> Scalar:
        ":returns: A scalar property as the magnitude of the vector."
        return Scalar(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + "_magnitude",
            mask=self.mask,
            func=lambda x: np.linalg.norm(x, axis=-1),
        )

    @property
    def log_magnitude(self) -> Scalar:
        ":returns: A scalar property as the log-magnitude of the vector."
        return Scalar(
            data_name=self.data_name,
            output_name=self.output_name + "_log10",
            mask=self.mask,
            func=lambda x: np.log10(np.linalg.norm(x, axis=-1)),
        )


@dataclass
class Matrix(Property):
    """Represent a matrix property of a dataset.

    Matrix properties should contain either 4 (2D) or 6 (3D) components.
    Matrix components can be accesses with brackets e.g. stress[0]
    """

    def __getitem__(self, index: int) -> Scalar:
        """
        Get a scalar property as a specific component of the matrix property.

        :param index: The index of the component.

        :returns: A scalar property as a matrix component.
        """
        suffix = {False: index, True: ["x", "y", "z", "xy", "yz", "xz"][index]}
        return Scalar(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + f"_{suffix[0 <= index <= 5]}",
            mask=self.mask,
            func=lambda x: np.array(x)[..., index],
            bilinear_cmap=True,
        )

    @property
    def magnitude(self) -> Scalar:
        ":returns: A scalar property as the frobenius norm of the matrix."
        return Scalar(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + "_magnitude",
            mask=self.mask,
            func=lambda x: np.linalg.norm(sym_tensor_to_mat(x), axis=(-2, -1)),
        )

    @property
    def trace(self) -> Scalar:
        ":returns: A scalar property as the trace of the matrix."
        return Scalar(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + "_trace",
            mask=self.mask,
            func=trace,
        )
