"""Defines the Scalar, Vector and Matrix Property classes.

They serve as classes to handle common physical properties in a systematic
way (e.g. temperature, pressure, displacement, …). Unit conversion is handled
via pint.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Callable, Union

import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap

from .custom_colormaps import mask_cmap
from .tensor_math import identity
from .unit_registry import u_reg


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
    func: Callable = identity
    """The function to be applied on the data.
       .. seealso:: :meth:`~ogstools.propertylib.Property.transform`"""
    mesh_dependent: bool = False
    """If the function to be applied is dependent on the mesh itself"""
    process_with_units: bool = False
    """If true, apply the function on values with units."""
    cmap: Union[Colormap, str] = "coolwarm"
    """Colormap to use for plotting."""
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

    @classmethod
    def from_property(cls, new_property: "Property", **changes):
        "Create a new Property object with modified attributes."
        return cls(
            data_name=new_property.data_name,
            data_unit=new_property.data_unit,
            output_unit=new_property.output_unit,
            output_name=new_property.output_name,
            mask=new_property.mask,
            func=new_property.func,
            mesh_dependent=new_property.mesh_dependent,
            process_with_units=new_property.process_with_units,
            cmap=new_property.cmap,
            categoric=new_property.categoric,
        ).replace(**changes)

    def transform(
        self,
        data: Union[int, float, np.ndarray, pv.DataSet, Sequence],
        strip_unit: bool = True,
    ) -> np.ndarray:
        """
        Return the transformed data values.

        Converts the data from data_unit to output_unit and apply a function
        the transformation function of this property. The result is returned by
        default without units. if `strip_unit` is False, a quantity is returned.

        Note:
        If `self.mesh_dependent` is True, `self.func` is applied directly to the
        DataSet. Otherwise, it is determined by `self.process_with_units` if the
        data is passed to the function with units (i.e. as a pint quantity) or
        without.
        """
        Qty, d_u, o_u = u_reg.Quantity, self.data_unit, self.output_unit
        if self.mesh_dependent:
            if isinstance(data, pv.DataSet):
                result = Qty(self.func(data, self), o_u)
            else:
                msg = "This property can only be evaluated on a mesh."
                raise TypeError(msg)
        else:
            if isinstance(data, pv.DataSet):
                result = Qty(self.func(Qty(self._get_data(data), d_u)), o_u)
            elif self.process_with_units:
                result = Qty(self.func(Qty(data, d_u)), o_u)
            else:
                result = Qty(Qty(self.func(np.asarray(data)), d_u), o_u)
        return result.magnitude if strip_unit else result

    def get_output_unit(self) -> str:
        """
        Get the output unit.

        returns: The output unit.
        """
        return "%" if self.output_unit == "percent" else self.output_unit

    @property
    def delta(self) -> "Property":
        "returns: A property relating to the difference in a quantity."
        data_property = self.replace(output_unit=self.data_unit)
        diff_unit = str(
            (
                data_property.transform(1, strip_unit=False)
                - data_property.transform(1, strip_unit=False)
            ).units
        )
        return self.replace(
            data_unit=diff_unit,
            output_unit=diff_unit,
            output_name=self.output_name + " difference",
            bilinear_cmap=True,
            cmap=self.cmap if self.bilinear_cmap else "coolwarm",
        )

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
        return Property(
            data_name=self.mask, mask=self.mask, categoric=True, cmap=mask_cmap
        )

    @property
    def magnitude(self) -> "Property":
        return self

    def mask_used(self, mesh: pv.UnstructuredGrid) -> bool:
        "Check whether the mesh contains the mask of this property."
        return (
            not self.is_mask()
            and self.mask in mesh.cell_data
            and (len(mesh.cell_data[self.mask]) != 0)
        )

    def _get_data(
        self, mesh: pv.UnstructuredGrid, masked: bool = True
    ) -> pv.UnstructuredGrid:
        """Get the data associated with a scalar or vector property from a mesh."""
        if (
            self.data_name not in mesh.point_data
            and self.data_name not in mesh.cell_data
        ):
            msg = f"Property {self.data_name} not found in mesh."
            raise IndexError(msg)
        if masked and self.mask_used(mesh):
            return mesh.ctp(True).threshold(value=[1, 1], scalars=self.mask)[
                self.data_name
            ]
        return mesh[self.data_name]


@dataclass
class Scalar(Property):
    "Represent a scalar property of a dataset."
