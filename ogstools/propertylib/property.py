"""Defines the ScalarProperty, VectorProperty and MatrixProperty classes.

It serves as a base class to handle common physical properties in a systematic
way (e.g. temperature, pressure, displacement, …). Unit conversion is handled
via pint.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Union
from typing import Optional as Opt

import numpy as np
from pint import UnitRegistry
from pint.facets.plain import PlainQuantity

from . import _mathfuncs as mf
from .utils import identity

u_reg = UnitRegistry()
u_reg.default_format = "~.12g"
u_reg.setup_matplotlib(True)


@dataclass
class ScalarProperty:
    """Represent a scalar property of a dataset."""

    data_name: str
    data_unit: str = ""
    output_unit: str = ""
    output_name: str = ""
    mask: Union[str, bool] = False
    func: Callable = identity

    def __post_init__(self):
        if not self.output_name:
            self.output_name = self.data_name

    @classmethod
    def from_function(cls, sfx: str, func: Callable, property):
        """
        Create a ScalarProperty object from a function.

        :param suffix: The suffix to be added to the output name.
        :param func: The function to be applied to the property.
        :param property: The original ScalarProperty object.

        :returns: The new ScalarProperty object.
        """
        return ScalarProperty(
            property.data_name,
            property.data_unit,
            property.output_unit,
            property.output_name + sfx,
            property.mask,
            func,
        )

    @classmethod
    def from_index(cls, id: int, property):
        """
        Create a ScalarProperty object from an index.

        :param index: The index to be used for component selection.
        :param property: The original ScalarProperty object.

        :returns: The new ScalarProperty object.
        """
        if 0 <= id <= 5:
            suffix = "_" + ["x", "y", "z", "xy", "yz", "xz"][id]
        else:
            suffix = f"_{id}"
        return cls.from_function(suffix, partial(mf.component, id=id), property)

    def __call__(
        self,
        data_name: Opt[str] = None,
        data_unit: Opt[str] = None,
        output_unit: Opt[str] = None,
        output_name: Opt[str] = None,
        mask: Opt[Union[str, bool]] = None,
        func: Opt[Callable] = None,
    ):
        """
        Create a new ScalarProperty object with modified attributes.

        :param data_name: The name of the data.
        :param data_unit: The unit of the data.
        :param output_unit: The unit of the output.
        :param output_name: The name of the output.
        :param mask: The name of the mask or if the property is a mask.
        :param func: The function to be applied.

        :returns: The new ScalarProperty object.
        """
        res = deepcopy(self)
        if data_name is not None:
            res.data_name = data_name
        if data_unit is not None:
            res.data_unit = data_unit
        if output_unit is not None:
            res.output_unit = output_unit
        if output_name is not None:
            res.output_name = output_name
        if mask is not None:
            res.mask = mask
        if func is not None:
            res.func = func
        return res

    def quantity(self, vals: np.ndarray) -> PlainQuantity:
        """
        Convert the values to a physical quantity with the output unit.

        :param vals: The input values.

        :returns: The values as a physical quantity.
        """
        return self.func(u_reg.Quantity(vals, self.data_unit)).to(
            self.output_unit
        )

    def cast(self, vals: np.ndarray) -> PlainQuantity:
        """
        Cast the values to a physical quantity with the output unit.

        :param vals: The input values.

        :returns: The values as a physical quantity.
        """
        return self.quantity(vals)

    def values(self, vals: np.ndarray) -> np.ndarray:
        """
        Get the values as a NumPy array without units.

        :param vals: The input values.

        :returns: The values without units.
        """
        return self.quantity(vals).magnitude

    def get_output_unit(self) -> str:
        """
        Get the output unit.

        returns: The output unit.
        """
        return "%" if self.output_unit == "percent" else self.output_unit

    def is_component(self) -> bool:
        """
        Check if the property is a component.

        :returns: True if the property is a component, False otherwise.
        """
        return isinstance(self.func, partial) and self.func.func == mf.component

    def is_mask(self) -> bool:
        """
        Check if the property is a component (using the outputname prefix).

        :returns: True if the property is a mask, False otherwise.
        """
        return isinstance(self.mask, bool) and self.mask


@dataclass
class VectorProperty(ScalarProperty):
    """Represents a vector property derived from a scalar property."""

    def __getitem__(self, index: int):
        """
        Get a scalar property as a specific component of the vector property.

        :param index: The index of the component.

        :returns: A scalar property representing the component.
        """
        return ScalarProperty.from_index(id=index, property=self)

    @property
    def component(self):
        """
        Property accessor for vector components.

        :returns: A partial function for creating scalar properties.
        """
        return partial(ScalarProperty.from_index, property=self)

    @property
    def magnitude(self):
        """
        Property accessor for the magnitude of the vector property.

        :returns: A scalar property representing the magnitude.
        """
        return ScalarProperty.from_function("_magnitude", mf.magnitude, self)

    @property
    def log_magnitude(self):
        """
        Property accessor for the logarithm of the magnitude.

        :returns: A scalar property representing the logarithm of the magnitude.
        """
        return ScalarProperty(
            self.data_name,
            "",
            "",
            self.output_name + "_log10",
            self.mask,
            mf.log_magnitude,
        )

    @property
    def trace(self):
        """
        Property accessor for the trace of the vector property.

        :returns: A scalar property representing the trace.
        """
        return ScalarProperty.from_function("_trace", mf.trace, self)


@dataclass
class MatrixProperty(ScalarProperty):
    """Represents a matrix property derived from a scalar property."""

    def __getitem__(self, index):
        """
        Get a scalar property as a specific component of the matrix property.

        :param index: The index of the component.

        :returns: A scalar property representing the component.
        """
        return ScalarProperty.from_index(id=index, property=self)

    @property
    def component(self):
        """
        Property accessor for matrix components.

        :returns: A partial function for creating scalar properties.
        """
        return partial(ScalarProperty.from_index, property=self)

    @property
    def trace(self):
        """
        Property accessor for the trace of the matrix property.

        :returns: The scalar property representing the trace.
        """
        return ScalarProperty.from_function("_trace", mf.trace, self)
