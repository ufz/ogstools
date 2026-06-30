# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from lxml import etree as ET

from ogstools.ogs6py import build_tree
from ogstools.property_types import PROPERTY_TYPES


class Media(build_tree.BuildTree):
    """
    Class for defining a media material properties."
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.media = self.populate_tree(self.root, "media", overwrite=True)

        self.properties: dict[str, list[str]] = {
            name: list(spec.parameters) for name, spec in PROPERTY_TYPES.items()
        }

    def _generate_generic_property(
        self, property_: ET.Element, args: dict[str, Any]
    ) -> None:
        for parameter in self.properties[args["type"]]:
            self.populate_tree(property_, parameter, text=args[parameter])

    def _generate_linear_property(
        self, property_: ET.Element, args: dict[str, Any]
    ) -> None:
        for parameter in self.properties[args["type"]]:
            self.populate_tree(property_, parameter, text=args[parameter])
        for var, param in args["independent_variables"].items():
            ind_var = self.populate_tree(property_, "independent_variable")
            self.populate_tree(ind_var, "variable_name", text=var)
            attributes = ["reference_condition", "slope"]
            for attrib in attributes:
                self.populate_tree(ind_var, attrib, text=str(param[attrib]))

    def _generate_function_property(
        self, property_: ET.Element, args: dict[str, Any]
    ) -> None:
        for parameter in self.properties[args["type"]]:
            value = self.populate_tree(
                property_, parameter, text=args[parameter]
            )
        self.populate_tree(value, "expression", text=args["expression"])
        for dvar in args["dvalues"]:
            dvalue = self.populate_tree(property_, "dvalue")
            self.populate_tree(dvalue, "variable_name", text=dvar)
            self.populate_tree(
                dvalue, "expression", text=args["dvalues"][dvar]["expression"]
            )

    def _generate_exponential_property(
        self, property_: ET.Element, args: dict[str, Any]
    ) -> None:
        for parameter in self.properties[args["type"]]:
            self.populate_tree(property_, parameter, text=args[parameter])
        exponent = self.populate_tree(property_, "exponent")
        self.populate_tree(
            exponent, "variable_name", text=args["exponent"]["variable_name"]
        )
        attributes = ["reference_condition", "factor"]
        for attrib in attributes:
            self.populate_tree(
                exponent, attrib, text=str(args["exponent"][attrib])
            )

    def _build_mpl_tree(self, args: dict) -> ET.Element:
        medium: ET.Element | None = None
        medium_id: str | None = args.get("medium_id")
        if "medium_id" not in args:
            args["medium_id"] = "0"

        if medium_id not in (None, "None"):
            medium = self.media.find(f"./medium[@id='{args['medium_id']}']")
        else:
            _media = self.media.findall("./medium")
            if len(_media) == 0:
                pass
            elif len(_media) > 1:
                msg = "Multiple media found but no id provided!"
                raise IndexError(msg)
            else:
                medium = _media[0]
                assert (
                    medium.attrib.get("id", "0") == "0"
                ), "Expected id='0' when no `medium_id` is given"

        if medium is None:
            medium = self.populate_tree(
                self.media, "medium", attr={"id": args["medium_id"]}
            )
        if "phase_type" in args:
            phases = self.get_child_tag(medium, "phases")
            if phases is None:
                phases = self.populate_tree(medium, "phases")
            phase = self.get_child_tag_for_type(
                phases, "phase", args["phase_type"]
            )
            if phase is None:
                phase = self.populate_tree(phases, "phase")
                self.populate_tree(phase, "type", text=args["phase_type"])
                if "component_name" in args:
                    components = self.populate_tree(phase, "components")
                    component = self.populate_tree(components, "component")
                    self.populate_tree(
                        component, "name", text=args["component_name"]
                    )
                    properties = self.populate_tree(component, "properties")
                else:
                    properties = self.populate_tree(phase, "properties")
            else:
                if "component_name" in args:
                    components = self.get_child_tag(phase, "components")
                    if components is None:
                        components = self.populate_tree(phase, "components")
                    component = self.get_child_tag_for_type(
                        components,
                        "component",
                        args["component_name"],
                        subtag="name",
                    )
                    if component is None:
                        component = self.populate_tree(components, "component")
                        self.populate_tree(
                            component, "name", text=args["component_name"]
                        )
                    properties = self.populate_tree(
                        component, "properties", overwrite=True
                    )
                else:
                    properties = self.get_child_tag(phase, "properties")
        else:
            properties = self.get_child_tag(medium, "properties")
            if properties is None:
                properties = self.populate_tree(medium, "properties")
        return properties

    def add_property(self, **args: Any) -> None:
        """
        Adds a property to medium/phase.

        Parameters
        ----------
        medium_id : `int` or `str`
        phase_type : `str` optional
        component_name : `str` optional
        name : `str`
        type : `str`
        value : `float` or `str`
        exponent : `float` or `str`
        cutoff_value : `float` or `str`
        independent_variable : `str`
        reference_condition : `float` or `str`
        reference_value : `float` or `str`
        slope : `float` or `str`
        parameter_name : `str`
        """
        self._convertargs(args)
        properties = self._build_mpl_tree(args)
        property_ = self.populate_tree(properties, "property")
        base_property_param = ["name", "type"]
        for param in base_property_param:
            self.populate_tree(property_, param, text=args[param])
        try:
            if args["type"] == "Linear":
                self._generate_linear_property(property_, args)
            elif args["type"] == "Exponential":
                self._generate_exponential_property(property_, args)
            elif args["type"] == "Function":
                self._generate_function_property(property_, args)
            else:
                self._generate_generic_property(property_, args)
        except KeyError:
            print("Material property parameters incomplete for")
            if "phase_type" in args:
                print(
                    f"Medium {args['medium_id']}->{args['phase_type']}->{args['name']}[{args['type']}]"
                )
            else:
                print(
                    f"Medium {args['medium_id']}->{args['name']}[{args['type']}]"
                )
