# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, TypeAlias

from lxml import etree as ET

OptionalETElement: TypeAlias = (
    ET.Element
)  # ToDo this should be Optional[ET.Element]


class BuildTree:
    """Helper class to create a nested dictionary
    representing the xml structure.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree

    @staticmethod
    def _convertargs(args: dict[str, Any]) -> None:
        for item, value in args.items():
            if not isinstance(value, list | dict):
                args[item] = str(value)

    @staticmethod
    def _to_str(text: Any) -> str:
        if isinstance(text, list):
            return " ".join(str(t) for t in text)
        if isinstance(text, bool):
            return str(text).lower()
        return str(text)

    @staticmethod
    def populate_tree(
        parent: ET.Element,
        tag: str,
        text: Any | list | dict | None = None,
        attr: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> ET.Element:
        """Add an element to the xml tree.

        :param parent:      Parent of the new element.
        :param tag:         Tag of the new element.
        :param text:        Text / value of the new element. If given a dict,
                            the key and value pairs will create corresponding
                            subelements.
        :param attr:        Attributes of the new element.
        :param overwrite:   If True, overrides the last child of parent with a
                            matching tag.
        """
        element = None
        if tag is not None:
            if overwrite:
                for child in parent:
                    if child.tag == tag:
                        element = child
            if element is None:
                element = ET.SubElement(parent, tag)
            if isinstance(text, dict):
                for key, value in text.items():
                    BuildTree.populate_tree(element, key, value)
            elif text is not None:
                element.text = BuildTree._to_str(text)
            if attr is not None:
                for key, val in attr.items():
                    element.set(key, str(val))
        return element

    @staticmethod
    def find_or_populate(parent: ET.Element, tag: str) -> ET.Element:
        "Find an existing tag in parent or create it."
        elem = parent.find(tag)
        if elem is None:
            elem = BuildTree.populate_tree(parent, tag)
        return elem

    @staticmethod
    def get_child_tag(
        parent: ET.Element,
        tag: str,
        attr: dict[str, str] | None = None,
        attr_val: str | None = None,
    ) -> OptionalETElement:
        """
        Search for child tag based on tag and possible attributes-
        """
        element = None
        for child in parent:
            if child.tag == tag:
                if not ((attr is None) and (attr_val is None)):
                    if child.get(attr) == attr_val:
                        element = child
                else:
                    element = child
        return element

    @staticmethod
    def get_child_tag_for_type(
        parent: ET.Element, tag: str, subtagval: str, subtag: str = "type"
    ) -> OptionalETElement:
        """
        Search for child tag based on subtag type.
        """
        element = None
        for child in parent:
            if child.tag == tag:
                for subchild in child:
                    if (subchild.tag == subtag) and (
                        subchild.text == subtagval
                    ):
                        element = child
        return element
