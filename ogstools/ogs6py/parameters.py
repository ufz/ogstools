# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from typing import Any

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

    def add_parameter(self, **args: Any) -> None:
        """
        Adds a parameter.

        Parameters
        ----------
        name : `str`
        type : `str`
        value : `float` or `str`
        values : `float` or `str`
        expression : `str`
        curve : `str`
        parameter : `str`
        mesh : `str`
        field_name : `str`
        time : `list`
        parameter_name : `list`
        use_local_coordinate_system : `bool` or `str`
        """
        self._convertargs(args)
        if "name" not in args:
            msg = "No parameter name given."
            raise KeyError(msg)
        if "type" not in args:
            msg = "Parameter type not given."
            raise KeyError(msg)
        parameter = self.populate_tree(self.parameters, "parameter")
        self.populate_tree(parameter, "name", text=args["name"])
        self.populate_tree(parameter, "type", text=args["type"])
        if args["type"] == "Constant":
            if "value" in args:
                self.populate_tree(parameter, "value", text=args["value"])
            elif "values" in args:
                self.populate_tree(parameter, "values", text=args["values"])
        elif args["type"] == "MeshElement" or args["type"] == "MeshNode":
            if "mesh" in args:
                self.populate_tree(parameter, "mesh", text=args["mesh"])
            self.populate_tree(parameter, "field_name", text=args["field_name"])
        elif args["type"] == "Function":
            if "mesh" in args:
                self.populate_tree(parameter, "mesh", text=args["mesh"])
            if isinstance(args["expression"], str) is True:
                self.populate_tree(
                    parameter, "expression", text=args["expression"]
                )
            elif isinstance(args["expression"], list) is True:
                for entry in args["expression"]:
                    self.populate_tree(parameter, "expression", text=entry)
        elif args["type"] == "CurveScaled":
            if "curve" in args:
                self.populate_tree(parameter, "curve", text=args["curve"])
            if "parameter" in args:
                self.populate_tree(
                    parameter, "parameter", text=args["parameter"]
                )
        elif args["type"] == "TimeDependentHeterogeneousParameter":
            if "time" not in args:
                msg = "time missing."
                raise KeyError(msg)
            if "parameter_name" not in args:
                msg = "Parameter name missing."
                raise KeyError(msg)
            if len(args["time"]) != len(args["parameter_name"]):
                msg = "parameter_name and time lists have different length."
                raise KeyError(msg)
            time_series = self.populate_tree(parameter, "time_series")
            for i, _ in enumerate(args["parameter_name"]):
                ts_pair = self.populate_tree(time_series, "pair")
                self.populate_tree(ts_pair, "time", text=str(args["time"][i]))
                self.populate_tree(
                    ts_pair, "parameter_name", text=args["parameter_name"][i]
                )
        else:
            msg = "Parameter type not supported (yet)."
            raise KeyError(msg)
        if ("use_local_coordinate_system" in args) and (
            (args["use_local_coordinate_system"] == "true")
            or (args["use_local_coordinate_system"] is True)
        ):
            self.populate_tree(
                parameter, "use_local_coordinate_system", text="true"
            )
