# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Defines the Scalar, Vector and Matrix Variable classes.

They serve as classes to handle common physical variables in a systematic
way (e.g. temperature, pressure, displacement, …). Unit conversion is handled
via pint.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap
from pint.facets.plain import PlainQuantity

from .custom_colormaps import mask_cmap
from .tensor_math import identity
from .unit_registry import u_reg


@dataclass
class Variable:
    """Represent a generic mesh variable."""

    data_name: str
    """The name of the variable data in the mesh."""
    data_unit: str = ""
    """The unit of the variable data in the mesh."""
    output_unit: str = ""
    """The output unit of the variable."""
    output_name: str = ""
    """The output name of the variable."""
    symbol: str = ""
    """The symbol representing this variable."""
    mask: str = ""
    """The name of the mask data in the mesh."""
    func: Callable = identity
    """The function to be applied on the data.
       .. seealso:: :meth:`~ogstools.variables.variable.Variable.transform`"""
    mesh_dependent: bool = False
    """If the function to be applied is dependent on the mesh itself"""
    process_with_units: bool = False
    """If true, apply the function on values with units."""
    cmap: Colormap | str = "coolwarm"
    """Colormap to use for plotting."""
    bilinear_cmap: bool = False
    """Should this variable be displayed with a bilinear cmap?"""
    categoric: bool = False
    """Does this variable only have categoric values?"""
    color: str | None = None
    """Default color for plotting"""

    def __post_init__(self) -> None:
        if not self.output_name:
            self.output_name = self.data_name

    @property
    def type_name(self) -> str:
        return type(self).__name__

    def replace(self: "Variable", **changes: Any) -> "Variable":
        """
        Create a new Variable object with modified attributes.

        Be aware that there is no type check safety here. So make sure, the new
        attributes and values are correct.

        :param changes: Attributes to be changed.

        :returns: A copy of the Variable with changed attributes.
        """
        return replace(self, **changes)

    @classmethod
    def from_variable(  # type: ignore[no-untyped-def]
        cls, new_variable: "Variable", **changes: Any
    ):
        "Create a new Variable object with modified attributes."
        return cls(
            data_name=new_variable.data_name,
            data_unit=new_variable.data_unit,
            output_unit=new_variable.output_unit,
            output_name=new_variable.output_name,
            symbol=new_variable.symbol,
            mask=new_variable.mask,
            func=new_variable.func,
            mesh_dependent=new_variable.mesh_dependent,
            process_with_units=new_variable.process_with_units,
            cmap=new_variable.cmap,
            bilinear_cmap=new_variable.bilinear_cmap,
            categoric=new_variable.categoric,
            color=new_variable.color,
        ).replace(**changes)

    def transform(
        self,
        data: int | float | np.ndarray | pv.UnstructuredGrid | Sequence,
        strip_unit: bool = True,
    ) -> np.ndarray:
        """
        Return the transformed data values.

        Converts the data from data_unit to output_unit and applies the
        transformation function of this variable. The result is returned by
        default without units. if `strip_unit` is False, a quantity is returned.

        Note:
        If `self.mesh_dependent` is True, `self.func` is applied directly to the
        mesh. Otherwise, it is determined by `self.process_with_units` if the
        data is passed to the function with units (i.e. as a pint quantity) or
        without.
        """
        Qty, d_u, o_u = u_reg.Quantity, self.data_unit, self.output_unit
        if self.mesh_dependent:
            if isinstance(data, pv.DataSet | pv.UnstructuredGrid):
                result = Qty(self.func(data, self), o_u)
            else:
                msg = "This variable can only be evaluated on a mesh."
                raise TypeError(msg)
        else:
            if isinstance(data, pv.DataSet | pv.UnstructuredGrid):
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
    def difference(self) -> "Variable":
        "returns: A variable relating to differences in this quantity."
        quantity = u_reg.Quantity(1, self.output_unit)
        diff_quantity: PlainQuantity = quantity - quantity
        diff_unit = str(diff_quantity.units)
        if str(diff_quantity.units) in ["degC", "°C"]:
            diff_unit = "kelvin"
        outname = self.output_name + "_difference"
        return self.replace(
            data_name=outname,
            data_unit=diff_unit,
            output_unit=diff_unit,
            output_name=outname,
            symbol=r"\Delta " + self.symbol,
            bilinear_cmap=True,
            func=identity,
            mesh_dependent=False,
            cmap=self.cmap if self.bilinear_cmap else "coolwarm",
        )

    def is_mask(self) -> bool:
        """
        Check if the variable is a mask.

        :returns: True if the variable is a mask, False otherwise.
        """
        return self.data_name == self.mask

    def get_mask(self) -> "Variable":
        """
        :returns: A variable representing this variables mask.
        """
        return Variable(
            data_name=self.mask, mask=self.mask, categoric=True, cmap=mask_cmap
        )

    @property
    def magnitude(self) -> "Variable":
        return self

    def mask_used(self, mesh: pv.UnstructuredGrid) -> bool:
        "Check whether the mesh contains the mask of this variable."
        return (
            not self.is_mask()
            and self.mask in mesh.cell_data
            and (len(mesh.cell_data[self.mask]) != 0)
        )

    def _get_data(
        self, mesh: pv.UnstructuredGrid, masked: bool = True
    ) -> np.ndarray:
        "Get the data associated with a scalar or vector variable from a mesh."
        if self.data_name not in (
            data_keys := ",".join(set().union(mesh.point_data, mesh.cell_data))
        ):
            msg = (
                f"Data name {self.data_name} not found in mesh. "
                f"Available data names are {data_keys}. "
            )
            raise KeyError(msg)
        if masked and self.mask_used(mesh):
            return mesh.ctp(True).threshold(value=[1, 1], scalars=self.mask)[
                self.data_name
            ]
        return mesh[self.data_name]

    def get_label(self, split_at: int | None = None) -> str:
        "Creates variable label in format 'variable_name / variable_unit'"
        unit_str = (
            f" / {self.get_output_unit()}" if self.get_output_unit() else ""
        )
        symbol_str = " " + f"${self.symbol}$" if self.symbol != "" else ""
        name = self.output_name
        if symbol_str != "":
            for suffix in ["xx", "yy", "zz", "yx", "yz", "xz", "x", "y", "z"]:
                if name.endswith(suffix):
                    name = name[: -(len(suffix) + 1)]
            for suffix in [str(num) for num in range(10)]:
                if name.endswith(str(suffix)):
                    name = name[:-2]
        label = name.replace("_", " ") + symbol_str + unit_str
        if split_at is None:
            return label
        return self._split_long_label(split_at, name, label)

    def _split_long_label(self, split_at: int, name: str, label: str) -> str:
        render_label = label.translate({ord(i): None for i in "{}$_^"})
        is_greek = False
        length = 0
        for c in render_label:
            if not is_greek:
                length += 1
            if is_greek and not c.isalpha():
                is_greek = False
                length += 1
            if c == "\\":
                is_greek = True
        if length >= split_at:
            try:
                split_index = min(
                    len(name), split_at - label[:split_at][::-1].index(" ")
                )
            except ValueError:
                split_index = len(name)
            label = label[0:split_index] + "\n" + label[split_index:]
        return label


@dataclass
class Scalar(Variable):
    "Represent a scalar variable."
