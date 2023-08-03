"""Defines the ScalarProperty, VectorProperty and MatrixProperty classes.

It serves as a base class to handle common physical properties in a systematic
way (e.g. temperature, pressure, displacement, …). Unit conversion is handled
via pint.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Literal, Union

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


class TagType(Enum):
    """Enum for property tag types."""

    mask = "mask"
    component = "component"
    unit_dim_const = "unit_dim_const"


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
    tag: Union[
        TagType, Literal["mask", "component", "unit_dim_const"], None
    ] = None
    """A tag to signify special meanings of the property."""

    def __post_init__(self):
        if not self.output_name:
            self.output_name = self.data_name
        if isinstance(self.tag, TagType) or self.tag is None:
            return
        tag_vals = [tag_type.value for tag_type in TagType]
        if self.tag not in tag_vals:
            msg = f"Unknown {self.tag=}. Allowed are {tag_vals}."
            raise ValueError(msg)

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
        if self.tag in [TagType.unit_dim_const, TagType.component]:
            return Q_(self.func(Q_(vals, _du).magnitude), _du).to(_ou)
        return Q_(self.func(Q_(vals, _du)), _ou)

    def values(self, vals: np.ndarray) -> np.ndarray:
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

    def is_component(self) -> bool:
        """
        Check if the property is a component (using tag).

        :returns: True if the property is a component, False otherwise.
        """
        return self.tag == TagType.component

    def is_mask(self) -> bool:
        """
        Check if the property is a mask (using tag).

        :returns: True if the property is a mask, False otherwise.
        """
        return self.tag == TagType.mask

    def get_mask(self):
        """
        :returns: A property representing this properties mask.
        """
        return Property(data_name=self.mask, tag=TagType.mask)


@dataclass
class ScalarProperty(Property):
    "Represent a scalar property of a dataset."


@dataclass
class VectorProperty(Property):
    """Represent a vector property of a dataset.

    Vector properties should contain either 2 (2D) or 3 (3D) components.
    """

    def __getitem__(self, index: int) -> ScalarProperty:
        """
        Get a scalar property as a specific component of the vector property.

        :param index: The index of the component.

        :returns: A scalar property as a vector component.
        """
        suffix = {False: index, True: ["x", "y", "z"][index]}
        return ScalarProperty(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + f"_{suffix[0 <= index <= 2]}",
            func=lambda x: np.array(x)[..., index],
            tag=TagType.component,
        )

    @property
    def magnitude(self) -> ScalarProperty:
        ":returns: A scalar property as the magnitude of the vector."
        return ScalarProperty(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + "_magnitude",
            func=lambda x: np.linalg.norm(x, axis=-1),
            tag=TagType.unit_dim_const,
        )

    @property
    def log_magnitude(self) -> ScalarProperty:
        ":returns: A scalar property as the log-magnitude of the vector."
        return ScalarProperty(
            data_name=self.data_name,
            output_name=self.output_name + "_log10",
            mask=self.mask,
            func=lambda x: np.log10(np.linalg.norm(x, axis=-1)),
            tag=TagType.unit_dim_const,
        )


@dataclass
class MatrixProperty(Property):
    """Represent a matrix property of a dataset.

    Matrix properties should contain either 4 (2D) or 6 (3D) components.
    """

    def __getitem__(self, index: int) -> ScalarProperty:
        """
        Get a scalar property as a specific component of the matrix property.

        :param index: The index of the component.

        :returns: A scalar property as a matrix component.
        """
        suffix = {False: index, True: ["x", "y", "z", "xy", "yz", "xz"][index]}
        return ScalarProperty(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + f"_{suffix[0 <= index <= 5]}",
            func=lambda x: np.array(x)[..., index],
            tag=TagType.component,
        )

    @property
    def magnitude(self) -> ScalarProperty:
        ":returns: A scalar property as the frobenius norm of the matrix."
        return ScalarProperty(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + "_magnitude",
            func=lambda x: np.linalg.norm(sym_tensor_to_mat(x), axis=(-2, -1)),
            tag=TagType.unit_dim_const,
        )

    @property
    def trace(self) -> ScalarProperty:
        ":returns: A scalar property as the trace of the matrix."
        return ScalarProperty(
            data_name=self.data_name,
            data_unit=self.data_unit,
            output_unit=self.output_unit,
            output_name=self.output_name + "_trace",
            func=trace,
            tag=TagType.unit_dim_const,
        )
