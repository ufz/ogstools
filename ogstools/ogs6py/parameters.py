# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal, overload

from lxml import etree as ET

from ogstools.ogs6py import build_tree


class Parameters(build_tree.BuildTree):
    """
    Class for managing the parameters section of the project file.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.parameters = self.populate_tree(
            self.root, "parameters", overwrite=True
        )

    def add_parameter(self, **kwargs: Any) -> None:
        """
        Adds a parameter.

        Parameters
        ----------
        name : `str`
            Name of the parameter.
        type : `str`
            Type of the parameter, one of `Constant`, `CurveScaled`, `Function`,
            `Group`, `MeshElement`, `MeshNode`, `RandomFieldMeshElement`,
            `Raster`, or `TimeDependentHeterogeneousParameter`.
        value : `float` or `str`
            Value for a constant parameter.
        values : `float` or `str`
            Values for a constant parameter.
        expression : `str` or `list[str]`
            Expression describing a function (valid for function parameter).
        curve : `str`
            Name of the curve (used in CurveScaled parameter).
        parameter : `str`
            Used in CurveScaled parameter; name of the parameter scaled by the
            curve.
        mesh : `str`
            Used in MeshElement or MeshNode parameter; specification of the mesh
            the parameter is defined on.
        field_name : `str`
            Used in MeshElement or MeshNode parameter; reference to the
            PropertyVector / DataArray given in the mesh.
        time : `list[float]`
            Used in TimeDependentHeterogeneousParameter.
        parameter_name : `list[str]`
            Used in `CurveScaled` to specify the parameter that shall be scaled.
        use_local_coordinate_system : `bool` or `str`

        Raises
        ------
        KeyError
            If 'name' or 'type' is not provided.
        KeyError
            If the parameter type is not supported.

        """
        self._convertargs(kwargs)
        match param_type := kwargs.pop("type"):
            case "Constant":
                param = self.add_constant_parameter(**kwargs)
            case "MeshNode" | "MeshElement":
                param = self.add_mesh_parameter(type=param_type, **kwargs)
            case "Function":
                param = self.add_function_parameter(**kwargs)
            case "CurveScaled":
                param = self.add_curve_scaled_parameter(**kwargs)
            case "TimeDependentHeterogeneousParameter":
                param = self.add_time_dependent_heterogeneous_parameter(
                    **kwargs
                )
            case _:
                msg = f"Parameter type '{param_type}' not supported (yet)."
                raise KeyError(msg)
        if kwargs.get("use_local_coordinate_system") in [True, "true"]:
            self.use_local_coordinate_system(param)

    def use_local_coordinate_system(self, parameter: ET.Element) -> None:
        "Add the local coordinate system element."
        self.populate_tree(
            parameter, "use_local_coordinate_system", text="true"
        )

    def _prepare_parameter(self, **kwargs: Any) -> ET.Element:
        param = self.populate_tree(self.parameters, "parameter")
        for key in ["name", "type"]:
            if key not in kwargs:
                msg = f"{key} not given."
                raise KeyError(msg)
            self.populate_tree(param, key, text=kwargs[key])
        return param

    @overload
    def add_constant_parameter(
        self, name: str, *, value: str
    ) -> ET.Element: ...

    @overload
    def add_constant_parameter(
        self, name: str, *, values: list[str]
    ) -> ET.Element: ...

    def add_constant_parameter(self, name: str, **kwargs: Any) -> ET.Element:
        """
        Add a constant parameter to the XML tree.

        :param name: parameter name

        Keyword Arguments:
            - value: str
            - values: list[str]
        """
        param = self._prepare_parameter(name=name, type="Constant")
        if "value" in kwargs:
            self.populate_tree(param, "value", text=kwargs["value"])
        elif "values" in kwargs:
            self.populate_tree(param, "values", text=kwargs["values"])
        else:
            msg = "Constant type parameter requires either a value or values."
            raise KeyError(msg)
        return param

    def add_mesh_parameter(
        self,
        name: str,
        type: Literal["MeshNode", "MeshElement"],
        field_name: str,
        mesh: str | None = None,
    ) -> ET.Element:
        """
        Add a mesh-based parameter (MeshElement or MeshNode) to the XML tree.

        :param name: parameter name
        :param type: parameter type
        :param field_name: fieldata name to read from the mesh
        :param mesh: mesh name
        """
        param = self._prepare_parameter(name=name, type=type)
        if mesh is not None:
            self.populate_tree(param, "mesh", text=mesh)
        self.populate_tree(param, "field_name", text=field_name)
        return param

    def add_function_parameter(
        self,
        name: str,
        expression: str | list[str],
        mesh: str | None = None,
    ) -> ET.Element:
        """
        Add a function parameter to the XML tree.

        :param name: parameter name
        :param expression: function expression of the parameter
        :param mesh: mesh name
        """
        param = self._prepare_parameter(name=name, type="Function")
        if mesh is not None:
            self.populate_tree(param, "mesh", text=mesh)
        if isinstance(expression, str):
            self.populate_tree(param, "expression", text=expression)
        elif isinstance(expression, list):
            for entry in expression:
                self.populate_tree(param, "expression", text=entry)
        return param

    def add_curve_scaled_parameter(
        self, name: str, curve: str, parameter: str
    ) -> ET.Element:
        """
        Add a curve-scaled parameter to the XML tree.

        :param name: parameter name
        :param curve: name of the curve which scales this parameter
        :param parameter: name of the parameter which is scaled
        """
        param = self._prepare_parameter(name=name, type="CurveScaled")
        self.populate_tree(param, "curve", text=curve)
        self.populate_tree(param, "parameter", text=parameter)
        return param

    def add_time_dependent_heterogeneous_parameter(
        self, name: str, time: list[Any], parameter_name: list[str]
    ) -> ET.Element:
        """
        Add a time-dependent heterogeneous parameter to the XML tree.

        :param name: parameter name
        :param time: list of timevalues
        :param parameter_name: list of parameter names
        """
        if len(time) != len(parameter_name):
            msg = "times and parameter_names have different lengths."
            raise ValueError(msg)

        param = self._prepare_parameter(
            name=name, type="TimeDependentHeterogeneousParameter"
        )
        time_series = self.populate_tree(param, "time_series")
        for time_val, param_name in zip(time, parameter_name, strict=True):
            ts_pair = self.populate_tree(time_series, "pair")
            self.populate_tree(ts_pair, "time", text=str(time_val))
            self.populate_tree(ts_pair, "parameter_name", text=param_name)
        return param
