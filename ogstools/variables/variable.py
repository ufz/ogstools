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
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap
from pint.facets.plain import PlainQuantity
from typing_extensions import Self

from .custom_colormaps import mask_cmap
from .func import Function
from .unit_registry import u_reg

Mesh: TypeAlias = pv.DataSet | pv.UnstructuredGrid
MeshOrSeries: TypeAlias = Mesh | Sequence[Mesh]
Data: TypeAlias = int | float | np.ndarray | MeshOrSeries

DERIVATIVE_SUFFIXES = ["gradient", "divergence", "vorticity", "qcriterion"]


@dataclass
class Variable:
    """Represent a generic mesh variable."""

    data_name: str
    """The name of the variable data in the mesh."""
    data_unit: str = ""
    """The unit of the variable data in the mesh."""
    output_unit: str = cast(str, None)
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
    spatial_unit: str = "m"
    """Default spatial unit"""
    temporal_unit: str = "s"
    """Default temporal unit"""

    def __post_init__(self, func: Function | Callable | None) -> None:
        self.output_unit = (
            self.data_unit if self.output_unit is None else self.output_unit
        )
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
    def from_variable(
        cls,
        variable: Variable,
        func: Callable | Function,
        suffix: str | int | None = None,
        prefix: str | int | None = None,
        **changes: Any,
    ) -> Self:
        """
        Create a new derived variable.

        `func` is appended to the stored functions. `prefix` and/or `suffix` are
        prepended to and appended to the base `output_name` (separated by "_");
        at least one has to be set. To change other attributes pass them
        as keyword arguments;

        :param variable: The variable to base the new one on.
        :param func: The function to append to the stored functions.
        :param prefix: Prepend this to the `output_name`.
        :param suffix: Append this to the `output_name`.
        :param changes: Other attributes to override in the new variable.
        :returns: A new Variable with the applied changes.
        :raises KeyError: If neither `prefix` nor `suffix` is set, or if
            ``output_name`` or ``functions`` are given in `changes`.
        """
        for k, v in {"output_name": "suffix", "functions": "func"}.items():
            if k in changes:
                msg = f"Please use `{v}` instead of `{k}`."
                raise KeyError(msg)
        if prefix is None and suffix is None:
            msg = "At least one of `prefix` or `suffix` has to be set."
            raise KeyError(msg)

        functions = variable.functions.copy()
        functions.append(func if isinstance(func, Function) else Function(func))
        output_name = "_".join(
            [
                str("" if prefix is None else prefix),
                variable.output_name or "",
                str("" if suffix is None else suffix),
            ]
        ).strip("_")
        return cls(
            data_name=variable.data_name,
            data_unit=variable.data_unit,
            output_unit=variable.output_unit,
            output_name=output_name,
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
        Return a Variable corresponding to `variable` for the given `data`.

        `data` may be a mesh or a MeshSeries. The name is resolved against
        predefined variables (`data_name` and `output_name`), mesh data and
        attributes of `data` (e.g. `points`, `timevalues`). "x", "y", "z" and
        "points*" resolve to the coordinates and "t", "time", "timevalues" to
        the time values; a trailing `_`-suffix (e.g. ``"displacement_0"``)
        selects a component of a vector/matrix variable.

        A Variable `variable` whose `output_name` is found in the data is
        returned reading that data directly (functions discarded); one
        whose ``data_name`` is found is returned unchanged. Otherwise a new
        Scalar, Vector or Matrix Variable is created depending on the shape
        of the data in the mesh.

        :param variable: The variable to retrieve or its name if a string.
        :param data: The mesh or mesh series containing the variable data.
        :returns: A matching predefined or new Variable.
        :raises KeyError: If the variable cannot be found in the data.
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
            if variable.data_name == "None":
                return variable
            if variable.output_name not in all_keys:
                if variable.data_name in data_keys:
                    return variable
                raise KeyError(error_msg)
            data_key = variable.output_name
            if variable.output_unit == "time_unit":
                variable = variable.replace(output_unit=_time_unit(data))

            if len(variable.functions) > 0:
                return variable.replace(
                    data_name=data_key,
                    data_unit=variable.output_unit,
                    output_unit=variable.output_unit,
                    output_name=data_key,
                    functions=[],
                )
            return variable.replace(data_name=data_key)

        # pylint: disable=import-outside-toplevel
        from ogstools.variables import all_variables

        # pylint: enable=import-outside-toplevel
        suffix = ""
        if (
            "_" in variable
            and variable not in all_keys
            and variable.rsplit("_", 1)[0] in all_keys
        ):
            parts = variable.rsplit("_", 1)
            variable, suffix = parts[0], parts[1] if len(parts) > 1 else ""

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

        if variable in data_keys:
            data_shape = mesh[variable].shape
        elif variable in dir(data):
            data_shape = getattr(getattr(mesh, variable), "shape", [0])
        else:
            raise KeyError(error_msg)
        if len(data_shape) == 1:
            return component(Scalar(variable), suffix)
        subclasses = Variable.__subclasses__()
        vector = next(x for x in subclasses if x.__name__ == "Vector")
        matrix = next(x for x in subclasses if x.__name__ == "Matrix")
        if data_shape[1] in [2, 3]:
            return component(vector(variable), suffix)
        return component(matrix(variable), suffix)

    def _get_input_values(
        self, data: Data, input: Callable | str | Variable
    ) -> np.ndarray | PlainQuantity:

        if isinstance(input, str):
            is_ms = isinstance(data, Sequence) and isinstance(data[0], Mesh)
            if isinstance(data, pv.DataSet) or is_ms:
                return data[input]  # type: ignore[reportReturnType, call-overload, index]
            msg = (
                f"{self.output_name} requires input {input}, thus it can only "
                "be applied on a mesh or a meshseries."
            )
            raise TypeError(msg)
        if isinstance(input, Variable):
            pwu = self.process_with_units
            unit = input.output_unit if pwu else input.data_unit
            var = input.replace(output_unit=unit, process_with_units=pwu)
            # to ensure, that this is only used on datasets, we call
            # _get_input_values again otherwise the same data will be
            # interpreted as different variables
            return var.transform(
                self._get_input_values(data, var.data_name), strip_unit=not pwu
            )
        return input(data)

    def _preprocess_params(self, params: dict) -> dict:
        return {
            p: (
                val.magnitude
                if isinstance(val, u_reg.Quantity)
                and not self.process_with_units
                else val
            )
            for p, val in params.items()
        }

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
        if self.output_unit == "time_unit":
            var = self.replace(output_unit=_time_unit(data))
            return var.transform(data, strip_unit)
        if (
            not u_reg.is_compatible_with(self.data_unit, self.output_unit)
            and not self.process_with_units
        ):
            msg = (
                "The data and output units are incompatible "
                f"({self.data_unit} -> {self.output_unit}). Please set "
                "`process_with_units` to True, to be sure the result of "
                "`transform` does not silently give false results."
            )
            raise TypeError(msg)
        is_ms = isinstance(data, Sequence) and isinstance(data[0], Mesh)
        is_dataset = isinstance(data, pv.DataSet) or is_ms
        result = self._get_data(data) if is_dataset else np.asarray(data)
        if self.process_with_units:
            result = u_reg.Quantity(result, self.data_unit)

        for function in self.functions:
            result = function.callable(
                result,
                *(self._get_input_values(data, inp) for inp in function.args),
                **self._preprocess_params(function.params),
            )

        if not self.process_with_units:
            result = u_reg.Quantity(result, self.data_unit)
        result = u_reg.Quantity(result, self.output_unit)
        return result.magnitude if strip_unit else result

    def _unit_convert(self, values: np.ndarray) -> np.ndarray:
        return u_reg.Quantity(
            u_reg.Quantity(values, self.data_unit), self.output_unit
        ).magnitude

    @property
    def get_output_unit(self) -> str:
        "Return the output unit"
        return "%" if self.output_unit == "percent" else self.output_unit

    @property
    def aggregations(self) -> int:
        "Returns the number of aggregating function stored in this Variable."
        return sum(f.aggregating for f in self.functions)

    def aggregation_variable(self, func: Callable) -> Self:
        "Returns an aggregating Variable corresponding to the function."
        name = func.__name__
        return getattr(
            self, name, self.aggregate(func, symbol=self.symbol, prefix=name)
        )

    def aggregate(self, func: Callable, **kwargs: Any) -> Self:
        """Creates a Variable aggregating the data with func."""
        subclasses = Variable.__subclasses__()
        vector = next(x for x in subclasses if x.__name__ == "Vector")
        matrix = next(x for x in subclasses if x.__name__ == "Matrix")
        index = -2 if isinstance(self, vector | matrix) else -1
        kwargs.setdefault(
            "func", Function(func, params={"axis": index}, aggregating=True)
        )
        if "suffix" not in kwargs:
            kwargs.setdefault("prefix", func.__name__)
        agg_var = type(self).from_variable(self, **kwargs)
        if "process_with_units" in kwargs:
            agg_var.functions.insert(-1, Function(self._unit_convert))
        return agg_var

    def normalize_aggregation(self, func: Callable | None = None) -> Self:
        """Returns a variable with exactly one aggregation function.

        If all previous functions plus the optionally given one contain not
        exactly one aggregating function an Error is thrown.
        """
        if func is not None:
            if self.aggregations > 0:
                msg = (
                    "Already got an aggregating variable. `func` may only be "
                    "passed if the variable is not aggregating itself."
                )
                raise TypeError(msg)
            return self.aggregation_variable(func)

        if self.aggregations != 1:
            msg = (
                "`variable` does not specify a single aggregation. Pass "
                "either an aggregating Variable with exactly one reduction "
                "(e.g. .min, .max, .mean, .median, .sum, .std, .var) or a "
                "variable together with a function, e.g. "
                "aggregate_temporal(variable, np.max)."
            )
            raise TypeError(msg)
        return self

    def change_axis(self, axis: int) -> Self:
        "Variable with an updated axis of its last aggregation function."
        var = self.copy()
        if var.aggregations == 0:
            msg = "This variable has no aggregation function."
            raise TypeError(msg)
        agg_func = next(f for f in var.functions[::-1] if f.aggregating)
        agg_func.params["axis"] = axis
        return var

    @property
    def min(self) -> Self:
        "A variable relating to the minimum of this quantity."
        return self.aggregate(
            np.min, symbol=f"{self.symbol}^{{min}}", prefix="min"
        )

    @property
    def max(self) -> Self:
        "A variable relating to the maximum of this quantity."
        return self.aggregate(
            np.max, symbol=f"{self.symbol}^{{max}}", prefix="max"
        )

    @property
    def mean(self) -> Self:
        "A variable relating to the mean of this quantity."
        return self.aggregate(
            np.mean, symbol=rf"\overline{{{self.symbol}}}", prefix="mean"
        )

    @property
    def median(self) -> Self:
        "A variable relating to the median of this quantity."
        return self.aggregate(
            np.median, symbol=rf"med({self.symbol})", prefix="median"
        )

    @property
    def sum(self) -> Self:
        "A variable relating to the sum of this quantity."
        return self.aggregate(
            np.sum,
            symbol=rf"\sum{{{self.symbol}}}",
            suffix="sum",
            process_with_units=True,
        )

    @property
    def std(self) -> Self:
        "A variable relating to the standard deviation of this quantity."
        return self.aggregate(
            np.std,
            symbol=f"SD({self.symbol})",
            suffix="standard_deviation",
            process_with_units=True,
            bilinear_cmap=False,
            cmap="coolwarm",
        )

    @property
    def var(self) -> Self:
        "A variable relating to the variance of this quantity."

        def square_unit(unit: str) -> str:
            return "" if unit == "" else unit + "**2"

        output_unit = (
            "" if self.output_unit in ["%", "percent"] else self.output_unit
        )

        return self.replace(output_unit=output_unit).aggregate(
            np.var,
            symbol=f"Var({self.symbol})",
            output_unit=square_unit(output_unit),
            suffix="variance",
            process_with_units=True,
            bilinear_cmap=False,
            cmap="coolwarm",
        )

    @property
    def argmin(self) -> Self:
        "A variable relating to the minimum index of this quantity."
        symbol = f"index({self.symbol}^{{min}})"
        return self.aggregate(
            np.argmin,
            symbol=symbol,
            prefix="min",
            suffix="index",
            output_unit="",
            process_with_units=True,
        )

    @property
    def argmax(self) -> Self:
        "A variable relating to the maximum index of this quantity."
        symbol = f"index({self.symbol}^{{max}})"
        return self.aggregate(
            np.argmax,
            symbol=symbol,
            prefix="max",
            suffix="index",
            output_unit="",
            process_with_units=True,
        )

    def _time_of(self, extremum: Literal["min", "max"]) -> Self:
        agg_var = self.aggregate(
            getattr(np, f"arg{extremum}"),
            symbol=f"t({self.symbol}={self.symbol}^{{{extremum}}})",
            prefix="time_of",
            suffix=extremum,
            output_unit="time_unit",
            process_with_units=True,
        )
        time_func = Function(
            lambda ids, tv, data: u_reg.Quantity(tv[ids], _time_unit(data)),
            ["timevalues", lambda data: data],
        )
        agg_var.functions.append(time_func)
        return agg_var

    @property
    def time_of_min(self) -> Self:
        "A variable relating to the time of the minimum of this quantity."
        return self._time_of("min")

    @property
    def time_of_max(self) -> Self:
        "A variable relating to the time of the maximum of this quantity."
        return self._time_of("max")

    @property
    def abs(self) -> Self:
        "A variable relating to absolute value of this quantity."
        return type(self).from_variable(
            self, func=np.abs, prefix="absolute", symbol=rf"|{self.symbol}|"
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
            func=[],
            cmap=self.cmap if self.bilinear_cmap else "coolwarm",
        )

    def rate(self, time_unit: str = "s") -> Self:
        "A variable relating to rate change of this quantity."
        diff_unit = self._diff_unit(self.output_unit)
        rate_unit = f"{diff_unit or 1}/{time_unit}"

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
            func=Function(compute_rate, ["timevalues", "time_unit"]),
            suffix="rate",
            data_name=self.data_name,
            data_unit=rate_unit,
            output_unit=rate_unit,
            symbol=rf"\dot{{{self.symbol}}}",
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
            output_unit="%",
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

            def mask_vals(mesh: pv.UnstructuredGrid) -> np.ndarray:
                if self.data_name in mesh.point_data:
                    mesh_ = mesh.ctp(pass_cell_data=False)
                else:
                    mesh_ = mesh
                return np.asarray(mesh_[self.mask] == 0)

            mask0 = mask_vals(mesh0)
            if not isinstance(dataset, Sequence):
                values[mask0] = np.nan
                return values

            if np.all(dataset[self.mask] == mesh0[self.mask]):  # type: ignore[call-overload]
                # Masks are time-invariant
                values[:, mask0] = np.nan
                return values

            # Masks differ with time
            for i, _mesh in enumerate(dataset):
                values[i, mask_vals(_mesh)] = np.nan
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

    def _get_derivative(self, derivative_type: str = "gradient") -> Variable:

        def derivative(
            mesh: pv.UnstructuredGrid,
            variable: Variable,
            **kwargs: Any,
        ) -> np.ndarray:
            # pylint: disable=import-outside-toplevel
            from ogstools.mesh.derivatives import calculate_derivative

            # pylint: enable=import-outside-toplevel
            derivative_data = calculate_derivative(
                mesh, variable, return_data=True, **kwargs
            )
            assert isinstance(derivative_data, np.ndarray)
            return derivative_data

        # pylint: enable=import-outside-toplevel
        new_data_unit = f"{self.data_unit}/{self.spatial_unit}"
        new_output_unit = f"{self.output_unit}/{self.spatial_unit}"
        if derivative_type == "gradient":
            new_output_name = f"{self.output_name}_gradient"
            new_symbol = rf"\nabla {self.symbol}"
        elif derivative_type == "divergence":
            new_output_name = f"{self.output_name}_divergence"
            new_symbol = rf"\nabla\cdot {self.symbol}"
        elif derivative_type == "vorticity":
            new_output_name = f"{self.output_name}_vorticity"
            new_symbol = rf"\omega \nabla\cdot {self.symbol}"
        elif derivative_type == "qcriterion":
            new_output_name = f"{self.output_name}_qcriterion"
            new_symbol = rf"$Q$ {self.symbol}"
        else:
            msg = """Incorrect derivative type.\n
                     Only gradient, divergence, vorticity and qcriterion are supported."""
            raise ValueError(msg)

        return type(self).from_variable(
            self,
            data_name=self.data_name,
            data_unit=new_data_unit,
            output_unit=new_output_unit,
            output_name=new_output_name,
            symbol=new_symbol,
            mask=self.mask,
            func=lambda data: derivative(deepcopy(data), self),
            process_with_units=self.process_with_units,
            cmap=self.cmap,
            bilinear_cmap=self.bilinear_cmap,
            categoric=self.categoric,
            color=self.color,
        )

    @property
    def gradient(self) -> Variable:
        return self._get_derivative("gradient")

    @property
    def divergence(self) -> Variable:
        return self._get_derivative("divergence")

    @property
    def vorticity(self) -> Variable:
        return self._get_derivative("vorticity")

    @property
    def qcriterion(self) -> Variable:
        return self._get_derivative("qcriterion")

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


def _time_unit(data: Data) -> str:
    return _quantity_to_str(getattr(data, "time_unit", "s"))


def time_var(var_name: str, data: Data) -> Scalar:
    unit = _time_unit(data)
    return Scalar("timevalues", unit, unit, var_name, symbol="t")
