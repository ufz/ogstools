# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

"""Defines the Scalar, Vector and Matrix Variable classes.

They serve as classes to handle common physical variables in a systematic
way (e.g. temperature, pressure, displacement, …). Unit conversion is handled
via pint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import InitVar, dataclass, field, replace
from typing import Any, TypeAlias, cast

import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap
from pint.facets.plain import PlainQuantity
from typing_extensions import Self

from .custom_colormaps import mask_cmap
from .func import Function
from .tensor_math import identity
from .unit_registry import u_reg

Mesh: TypeAlias = pv.DataSet | pv.UnstructuredGrid
MeshOrSeries: TypeAlias = Mesh | Sequence[Mesh]
Data: TypeAlias = int | float | np.ndarray | MeshOrSeries


@dataclass
class Variable:
    """Represent a generic mesh variable."""

    data_name: str
    """The name of the variable data in the mesh."""
    data_unit: str = ""
    """The unit of the variable data in the mesh."""
    output_unit: str = ""
    """The output unit of the variable."""
    output_name: str = cast(str, None)
    """The output name of the variable."""
    symbol: str = ""
    """The symbol representing this variable."""
    mask: str = ""
    """The name of the mask data in the mesh."""
    func: InitVar[Function | Callable | None] = None
    """The function to be applied on the data."""
    functions: list[Function] = field(default_factory=list)
    """Contains this and all previous functions."""
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

    def __post_init__(self, func: Function | Callable | None) -> None:
        self.output_unit = self.output_unit or self.data_unit
        self.output_name = (
            self.data_name if self.output_name is None else self.output_name
        )
        if func is not None:
            self.function = func

    def __str__(self) -> str:
        return self.data_name

    @property
    def function(self) -> Function | None:
        """Returns the final function"""
        return self.functions[-1] if self.functions else None

    @function.setter
    def function(self, func: Callable | Function | list[Function]) -> None:
        """Set's this Variable's function.

        If given a list of functions, all stored functions are overwritten."""
        if isinstance(func, list):
            self.functions = func
            return
        new_func = func if isinstance(func, Function) else Function(func)
        if len(self.functions) == 0:
            self.functions = [new_func]
        else:
            self.functions[-1] = new_func

    @property
    def type_name(self) -> str:
        return type(self).__name__

    def replace(self, **changes: Any) -> Self:
        """
        Create a new Variable object with modified attributes.

        Be aware that there is no type check safety here. So make sure, the new
        attributes and values are correct.

        :param changes: Attributes to be changed.

        :returns: A copy of the Variable with changed attributes.
        """
        if not set(changes).issubset(set(dir(self))):
            wrong_keys = ", ".join(set(changes) - (set(dir(self))))
            msg = (
                "The following arguments are no attributes of "
                f"{type(self).__name__}: {wrong_keys}"
            )
            raise KeyError(msg)
        return replace(self.copy(), **changes)

    def copy(self, deep: bool = True) -> Self:
        if deep:
            return deepcopy(self)
        return copy(self)

    @classmethod
    def from_variable(cls, variable: Variable, **changes: Any) -> Self:
        "Create a new Variable object with modified attributes."
        functions = variable.functions.copy()
        if (func := changes.pop("func", None)) is not None:
            functions.append(
                func if isinstance(func, Function) else Function(func)
            )
        return cls(
            data_name=variable.data_name,
            data_unit=variable.data_unit,
            output_unit=variable.output_unit,
            output_name=variable.output_name,
            symbol=variable.symbol,
            mask=variable.mask,
            functions=functions,
            process_with_units=variable.process_with_units,
            cmap=variable.cmap,
            bilinear_cmap=variable.bilinear_cmap,
            categoric=variable.categoric,
            color=variable.color,
        ).replace(**changes)

    @classmethod
    def find(cls, variable: Variable | str, data: MeshOrSeries) -> Variable:
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
        mesh = data[0] if isinstance(data, Sequence) else data
        data_keys: list[str] = list(
            set().union(mesh.point_data, mesh.cell_data, mesh.field_data)
        )
        all_keys = data_keys + dir(data)
        error_msg = f"'{variable}' not found in dataset. Available data names are {data_keys}. "
        var_name = variable if isinstance(variable, str) else variable.data_name

        if var_name in ["x", "y", "z"] or var_name.startswith("points"):
            return spatial_var(var_name, data)
        if var_name in ["t", "time", "timevalues"]:
            return time_var(var_name, data)

        if isinstance(variable, Variable):
            if variable.data_name in all_keys + ["None"]:
                return variable
            matches = [
                variable.output_name in data_key for data_key in all_keys
            ]
            if not any(matches):
                raise KeyError(error_msg)
            data_key = all_keys[matches.index(True)]
            if data_key == variable.difference.output_name:
                return variable.difference
            if data_key in variable._agg_names:
                return variable.replace(
                    data_name=data_key,
                    data_unit=variable.output_unit,
                    output_unit=variable.output_unit,
                    output_name=data_key,
                    symbol=variable.symbol,
                    func=[Function(identity)],
                )
            return variable.replace(data_name=data_key, output_name=data_key)

        # pylint: disable=import-outside-toplevel
        from ogstools.variables import all_variables

        # pylint: enable=import-outside-toplevel
        suffix = ""
        if (
            "_" in variable
            and variable not in all_keys
            and variable.rsplit("_", 1)[0] in all_keys
        ):
            variable, suffix = variable.rsplit("_", 1)

        def component(var: Variable, suffix: str) -> Variable:
            suffix_ = int(suffix) if suffix.isdigit() else suffix
            if suffix == "":
                return var
            if isinstance(var, Scalar):
                msg = f"Scalar '{var.data_name}' has no component {suffix}."
                raise KeyError(msg)
            return var[suffix_]  # type: ignore[index]

        for prop in all_variables:
            if prop.data_name == variable:
                return component(prop, suffix)
        for prop in all_variables:
            if prop.output_name == variable and variable != "":
                if prop.data_name in all_keys:
                    return component(prop, suffix)
                if prop.output_name in all_keys:
                    return component(
                        prop.replace(data_name=prop.output_name), suffix
                    )

        if variable not in all_keys:
            raise KeyError(error_msg)

        if variable in dir(data):
            data_shape = getattr(mesh, variable).shape
        else:
            data_shape = mesh[variable].shape
        if len(data_shape) == 1:
            return component(Scalar(variable), suffix)
        subclasses = Variable.__subclasses__()
        vector = next(x for x in subclasses if x.__name__ == "Vector")
        matrix = next(x for x in subclasses if x.__name__ == "Matrix")
        if data_shape[1] in [2, 3]:
            return component(vector(variable), suffix)
        return component(matrix(variable), suffix)

    def _get_input_values(
        self, data: Data, input: Callable | str
    ) -> np.ndarray:

        if isinstance(input, str):
            is_ms = isinstance(data, Sequence) and isinstance(data[0], Mesh)
            if isinstance(data, pv.DataSet) or is_ms:
                return data[input]  # type: ignore[reportReturnType, call-overload, index]
            msg = (
                f"{self} requires input {input}, thus it can only be "
                "applied on a mesh or a meshseries."
            )
            raise TypeError(msg)
        return input(data)

    def transform(
        self, data: Data, strip_unit: bool = True
    ) -> np.ndarray | PlainQuantity:
        """
        Return the transformed data values.

        Converts the data from data_unit to output_unit and applies the
        transformation function of this variable. The result is returned by
        default without units. if `strip_unit` is False, a quantity is returned.

        Note:
        If applied on a mesh or a meshseries, the data_name is read from the
        dataset and passed to the stored function. If applied on numeric data,
        it is passed to the function.
        if `process_with_units` is True data is passed to the function with
        units (i.e. as a pint quantity).
        """
        is_ms = isinstance(data, Sequence) and isinstance(data[0], Mesh)
        is_dataset = isinstance(data, pv.DataSet) or is_ms
        result = self._get_data(data) if is_dataset else np.asarray(data)
        if self.process_with_units:
            result = u_reg.Quantity(result, self.data_unit)

        for function in self.functions:
            result = function.callable(
                result,
                *(self._get_input_values(data, inp) for inp in function.args),
                **function.params,
            )

        if not self.process_with_units:
            result = u_reg.Quantity(result, self.data_unit)
        result = u_reg.Quantity(result, self.output_unit)
        return result.magnitude if strip_unit else result

    @property
    def get_output_unit(self) -> str:
        "Return the output unit"
        return "%" if self.output_unit == "percent" else self.output_unit

    @property
    def _agg_names(self) -> list[str]:
        return [
            self.min.output_name,
            self.max.output_name,
            self.mean.output_name,
            self.median.output_name,
            self.sum.output_name,
            self.std.output_name,
            self.var.output_name,
        ]

    def _agg(self, func: Callable, new_symbol: str | None) -> Self:
        subclasses = Variable.__subclasses__()
        vector = next(x for x in subclasses if x.__name__ == "Vector")
        matrix = next(x for x in subclasses if x.__name__ == "Matrix")
        index = -2 if isinstance(self, vector | matrix) else -1
        return type(self).from_variable(
            self,
            func=lambda x: func(x, axis=index),
            output_name="_".join([self.output_name, func.__name__]),
            symbol=new_symbol,
        )

    @property
    def min(self) -> Self:
        "A variable relating to minimum of this quantity."
        return self._agg(np.min, f"{self.symbol}_{{min}}")

    @property
    def max(self) -> Self:
        "A variable relating to maximum of this quantity."
        return self._agg(np.max, f"{self.symbol}_{{max}}")

    @property
    def mean(self) -> Self:
        "A variable relating to mean of this quantity."
        return self._agg(np.mean, rf"\overline{{{self.symbol}}}")

    @property
    def median(self) -> Self:
        "A variable relating to median of this quantity."
        return self._agg(np.median, rf"med({self.symbol})")

    @property
    def sum(self) -> Self:
        "A variable relating to sum of this quantity."
        return self._agg(np.sum, rf"\sum{{{self.symbol}}}")

    @property
    def std(self) -> Self:
        "A variable relating to standard deviation of this quantity."
        return self._agg(np.std, f"SD({self.symbol})")

    @property
    def var(self) -> Self:
        "A variable relating to variance of this quantity."

        def square_unit(unit: str) -> str:
            return "" if unit == "" else unit + "**2"

        return self._agg(np.var, f"Var({self.symbol})").replace(
            data_unit=square_unit(self.data_unit),
            output_unit=square_unit(self.output_unit),
        )

    @property
    def abs(self) -> Self:
        "A variable relating to absolute value of this quantity."
        return type(self).from_variable(
            self,
            output_name=f"absolute_{self.output_name}",
            symbol=rf"|{self.symbol}|",
            func=np.abs,
        )

    def _diff_unit(self, unit: str) -> str:
        quantity = u_reg.Quantity(1, unit)
        diff_quantity: PlainQuantity = quantity - quantity
        diff_unit = str(diff_quantity.units)
        if str(diff_quantity.units) in ["degC", "°C"]:
            diff_unit = "kelvin"
        return diff_unit

    @property
    def difference(self) -> Variable:
        "A variable relating to differences in this quantity."
        diff_unit = self._diff_unit(self.output_unit)
        outname = self.output_name + "_difference"
        return self.replace(
            data_name=outname,
            data_unit=diff_unit,
            output_unit=diff_unit,
            output_name=outname,
            symbol=r"\Delta " + self.symbol,
            bilinear_cmap=True,
            func=[Function(identity)],
            cmap=self.cmap if self.bilinear_cmap else "coolwarm",
        )

    def rate(self, time_unit: str = "s") -> Self:
        "A variable relating to rate change of this quantity."
        diff_unit = self._diff_unit(self.output_unit)
        rate_unit = f"{diff_unit or 1}/{time_unit}"
        outname = self.output_name + "_rate"

        def compute_rate(
            values: np.ndarray, timevalues: np.ndarray, data_time_unit: str
        ) -> np.ndarray:
            factor = u_reg.Quantity(data_time_unit).to(time_unit).magnitude
            delta = np.diff(values, axis=0, prepend=np.nan)
            # The following is required for numpy to correctly broadcast for
            # scalar and vector/matrix inputs
            dt_dim_expansion = (slice(None),) + (None,) * (len(delta.shape) - 1)
            dt = np.diff(timevalues * factor, prepend=1)[dt_dim_expansion]
            return delta / dt

        return type(self).from_variable(
            self,
            data_name=self.data_name,
            data_unit=rate_unit,
            output_unit=rate_unit,
            output_name=outname,
            symbol=rf"\dot{{{self.symbol}}}",
            func=Function(compute_rate, ["timevalues", "time_unit"]),
            bilinear_cmap=True,
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
        mask_data = next(  # type: ignore[call-overload]
            (d for d in [mesh.point_data, mesh.cell_data] if self.mask in d), {}
        ).get(self.mask, [])
        return (
            not self.is_mask()
            and (len(mask_data) != 0)
            and not np.all(mask_data == 1)
        )

    def _get_data(
        self,
        dataset: pv.UnstructuredGrid | Sequence,
        masked: bool = True,
    ) -> np.ndarray:
        "Get the data associated with a scalar or vector variable from a mesh."
        mesh0 = dataset[0] if isinstance(dataset, Sequence) else dataset
        if self.data_name not in (
            data_keys := set().union(
                mesh0.point_data, mesh0.cell_data, mesh0.field_data
            )
        ):
            for data in [mesh0, dataset]:
                if hasattr(data, self.data_name):
                    return getattr(data, self.data_name)
            if self.data_name in ["MaterialIDs", "None"]:
                return np.full(mesh0.number_of_cells, 0)
            msg = (
                f"Data name '{self.data_name}' not found in mesh. "
                f"Available data names are {', '.join(data_keys)}. "
            )
            raise KeyError(msg)

        values = dataset[self.data_name]  # type: ignore[call-overload]
        if masked and self.mask_used(dataset):
            mask0 = np.asarray(mesh0.ctp(pass_cell_data=False)[self.mask] == 0)
            if not isinstance(dataset, Sequence):
                values[mask0] = np.nan
                return values

            if np.all(dataset[self.mask] == mesh0[self.mask]):  # type: ignore[call-overload]
                # Masks are time-invariant
                values[:, mask0] = np.nan
                return values

            # Masks differ with time
            for i, _mesh in enumerate(dataset):
                mask = np.asarray(
                    _mesh.ctp(pass_cell_data=False)[self.mask] == 0
                )
                values[i, mask] = np.nan
        return values

    def get_label(self, split_at: int | None = None) -> str:
        "Creates variable label in format 'variable_name / variable_unit'"
        unit_str = f" / {self.get_output_unit}" if self.get_output_unit else ""
        symbol_str = " " + f"${self.symbol}$" if self.symbol != "" else ""
        name = self.output_name
        if symbol_str != "":
            cartesian_suf = ["xx", "yy", "zz", "yx", "yz", "xz", "x", "y", "z"]
            polar_suf = ["rr", "tt", "pp", "rt", "tp", "rp"]
            for suffix in cartesian_suf + polar_suf:
                if name.endswith(("_" + suffix, " " + suffix)):
                    name = name[: -(len(suffix) + 1)]
            for suffix in [str(num) for num in range(10)]:
                if name.endswith(("_" + str(suffix), " " + str(suffix))):
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


def _quantity_to_str(unit: PlainQuantity | str) -> str:
    if isinstance(unit, PlainQuantity):
        return str(unit if unit.magnitude != 1 else unit.units)
    return unit


def spatial_var(var_name: str, data: Data) -> Variable:

    unit = _quantity_to_str(getattr(data, "spatial_unit", "m"))

    from .vector import Vector

    pts_var = Vector("points", unit, unit, "", color="k")
    if var_name == "points":
        return pts_var
    if ("_" in var_name) and (suffix := var_name.rsplit("_", 1)[1]):
        return pts_var[suffix]  # type: ignore[index]
    return pts_var[var_name]  # type: ignore[index]


def time_var(var_name: str, data: Data) -> Scalar:
    unit = _quantity_to_str(getattr(data, "time_unit", "s"))
    return Scalar("timevalues", unit, unit, var_name, symbol="t")
