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

from __future__ import annotations

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

    def __init__(
        self,
        data_name: str,
        data_unit: str = "",
        output_unit: str | None = None,
        output_name: str | None = None,
        symbol: str = "",
        mask: str = "",
        func: Callable = identity,
        mesh_dependent: bool = False,
        process_with_units: bool = False,
        cmap: Colormap | str = "coolwarm",
        bilinear_cmap: bool = False,
        categoric: bool = False,
        color: str | None = None,
    ) -> None:
        self.data_name = data_name
        self.data_unit = data_unit
        self.output_unit = (
            str(output_unit) if output_unit is not None else data_unit
        )
        self.output_name = (
            str(output_name) if output_name is not None else data_name
        )
        self.symbol = symbol
        self.mask = mask
        self.func = func
        self.mesh_dependent = mesh_dependent
        self.process_with_units = process_with_units
        self.cmap = cmap
        self.bilinear_cmap = bilinear_cmap
        self.categoric = categoric
        self.color = color

    @property
    def type_name(self) -> str:
        return type(self).__name__

    def replace(self: Variable, **changes: Any) -> Variable:
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
        cls, new_variable: Variable, **changes: Any
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

    @classmethod
    def find(cls, variable: Variable | str, mesh: pv.DataSet) -> Variable:
        """
        Returns a Variable preset or creates one with correct type.

        Searches for presets by data_name and output_name and returns if found.
        If 'variable' is given as type Variable this will also look for
        derived variables (difference, aggregate).
        Otherwise create Scalar, Vector, or Matrix Variable depending on the shape
        of data in mesh.

        :param variable:    The variable to retrieve or its name if a string.
        :param mesh:        The mesh containing the variable data.
        :returns: A corresponding Variable preset or a new Variable of correct type.
        """
        data_keys: list[str] = list(
            set().union(mesh.point_data, mesh.cell_data)
        )
        error_msg = (
            f"Data not found in mesh. Available data names are {data_keys}. "
        )
        if isinstance(variable, str) and variable in ["x", "y", "z"]:
            return _spatial_preset(variable)
        if variable == "time":
            return _time_preset()

        if isinstance(variable, Variable):
            if variable.data_name in data_keys:
                return variable
            matches = [
                variable.output_name in data_key for data_key in data_keys
            ]
            if not any(matches):
                raise KeyError(error_msg)
            data_key = data_keys[matches.index(True)]
            # TODO: remove these here and return them from compute functions
            if data_key == f"{variable.output_name}_difference":
                return variable.difference
            if data_key.rsplit("_")[0] in [
                "min", "max", "mean", "median", "sum", "std", "var"  # fmt:skip
            ]:
                return variable.replace(
                    data_name=data_key,
                    data_unit=variable.output_unit,
                    output_unit=variable.output_unit,
                    output_name=data_key,
                    symbol=variable.symbol,
                    func=identity,
                    mesh_dependent=False,
                )
            return variable.replace(data_name=data_key, output_name=data_key)

        # pylint: disable=import-outside-toplevel
        from ogstools.variables import all_variables

        # pylint: enable=import-outside-toplevel

        for prop in all_variables:
            if prop.data_name == variable:
                return prop
        for prop in all_variables:
            if prop.output_name == variable:
                if prop.data_name in data_keys:
                    return prop
                return prop.replace(data_name=prop.output_name)

        matches = [variable in data_key for data_key in data_keys]
        if not any(matches):
            raise KeyError(error_msg)

        data_shape = mesh[variable].shape
        if len(data_shape) == 1:
            return Scalar(variable)
        subclasses = Variable.__subclasses__()
        vector = next(x for x in subclasses if x.__name__ == "Vector")
        matrix = next(x for x in subclasses if x.__name__ == "Matrix")
        if data_shape[1] in [2, 3]:
            return vector(variable)
        return matrix(variable)

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
        is_ms = isinstance(data, Sequence) and isinstance(data[0], pv.DataSet)
        if self.mesh_dependent:
            if isinstance(data, pv.DataSet | pv.UnstructuredGrid) or is_ms:
                result = Qty(self.func(data, self), o_u)
            else:
                msg = "This variable can only be evaluated on a mesh."
                raise TypeError(msg)
        else:
            if isinstance(data, pv.DataSet | pv.UnstructuredGrid) or is_ms:
                result = Qty(self.func(Qty(self._get_data(data), d_u)), o_u)
            elif self.process_with_units:
                result = Qty(self.func(Qty(data, d_u)), o_u)
            else:
                result = Qty(Qty(self.func(np.asarray(data)), d_u), o_u)
        return result.magnitude if strip_unit else result

    @property
    def get_output_unit(self) -> str:
        "Return the output unit"
        return "%" if self.output_unit == "percent" else self.output_unit

    @property
    def difference(self) -> Variable:
        "A variable relating to differences in this quantity."
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

    @property
    def abs_error(self) -> Variable:
        "A variable relating to an absolute error of this quantity."
        return self.difference.replace(
            data_name=f"{self.data_name}_abs_error",
            output_name="absolute_error",
            symbol="\\epsilon_\\mathrm{abs}",
            cmap="RdGy",
            bilinear_cmap=True,
        )

    @property
    def rel_error(self) -> Variable:
        "A variable relating to a relative error of this quantity."
        return self.difference.replace(
            data_name=f"{self.data_name}_rel_error",
            data_unit="",
            output_unit="percent",
            output_name="relative_error",
            symbol="\\epsilon_\\mathrm{rel}",
            cmap="PuOr",
            bilinear_cmap=True,
        )

    @property
    def anasol(self) -> Variable:
        "A variable relating to an analytical solution of this quantity."
        return self.replace(
            data_name=f"{self.data_name}_anasol",
            output_name=f"analytical {self.output_name} solution",
        )

    def is_mask(self) -> bool:
        """
        Check if the variable is a mask.

        :returns: True if the variable is a mask, False otherwise.
        """
        return self.data_name == self.mask

    def get_mask(self) -> Variable:
        "A variable representing this variables mask."
        return Variable(
            data_name=self.mask, mask=self.mask, categoric=True, cmap=mask_cmap
        )

    @property
    def magnitude(self) -> Variable:
        return self

    def mask_used(self, mesh: pv.UnstructuredGrid) -> bool:
        "Check whether the mesh contains the mask of this variable."
        return (
            not self.is_mask()
            and self.mask in mesh.cell_data
            and (len(mesh.cell_data[self.mask]) != 0)
        )

    def _get_data(
        self,
        dataset: pv.UnstructuredGrid | Sequence,
        masked: bool = True,
    ) -> np.ndarray:
        "Get the data associated with a scalar or vector variable from a mesh."
        mesh = dataset[0] if isinstance(dataset, Sequence) else dataset
        if self.data_name not in (
            data_keys := set().union(mesh.point_data, mesh.cell_data)
        ):
            msg = (
                f"Data name {self.data_name} not found in mesh. "
                f"Available data names are {', '.join(data_keys)}. "
            )
            raise KeyError(msg)
        if masked and self.mask_used(dataset):
            if isinstance(dataset, Sequence):
                mask = np.asarray(mesh.ctp(False)[self.mask] == 1)
                return dataset[self.data_name][:, mask]  # type: ignore[call-overload]
            return dataset.ctp(False).threshold(
                value=[1, 1], scalars=self.mask
            )[self.data_name]
        return dataset[self.data_name]  # type: ignore[call-overload]

    def get_label(self, split_at: int | None = None) -> str:
        "Creates variable label in format 'variable_name / variable_unit'"
        unit_str = f" / {self.get_output_unit}" if self.get_output_unit else ""
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


class Scalar(Variable):
    "Represent a scalar variable."


def _spatial_preset(axis: str) -> Scalar:
    # pylint: disable=import-outside-toplevel
    # Importing here dynamically to avoid circular import
    # If we want to avoid this, we'd have to move plot.setup to someplace
    # outside of plot
    from ogstools.plot import setup

    # pylint: enable=import-outside-toplevel

    def get_pts(
        index: int,
    ) -> Callable[[pv.UnstructuredGrid | Sequence, Variable], np.ndarray]:
        "Returns the coordinates of all points with the given index"

        def get_pts_coordinate(
            dataset: pv.UnstructuredGrid | Sequence, _: Variable
        ) -> np.ndarray:
            mesh = dataset[0] if isinstance(dataset, Sequence) else dataset
            return mesh.points[:, index]

        return get_pts_coordinate

    return Scalar(
        axis,
        setup.spatial_unit,  # type:ignore[attr-defined]
        setup.spatial_unit,  # type:ignore[attr-defined]
        mesh_dependent=True,
        func=get_pts("xyz".index(axis)),
        color="k",
    )


def _time_preset() -> Scalar:
    # pylint: disable=import-outside-toplevel
    # Importing here dynamically to avoid circular import
    # If we want to avoid this, we'd have to move plot.setup to someplace
    # outside of plot
    from ogstools.plot import setup

    # pylint: enable=import-outside-toplevel

    return Scalar(
        "time",
        setup.time_unit,  # type:ignore[attr-defined]
        setup.time_unit,  # type:ignore[attr-defined]
        mesh_dependent=True,
        func=lambda ms, _: getattr(ms, "timevalues", range(len(ms))),
    )
