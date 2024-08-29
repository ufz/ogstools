"""
Copyright (c) 2012-2021, OpenGeoSys Community (http://www.opengeosys.org)
            Distributed under a Modified BSD License.
              See accompanying file LICENSE or
              http://www.opengeosys.org/project/license

"""


from typing import TypeAlias

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
    def _convertargs(args: dict[str, str]) -> None:
        for item, value in args.items():
            if not isinstance(value, list | dict):
                args[item] = str(value)

    @staticmethod
    def populate_tree(
        parent: ET.Element,
        tag: str,
        text: str | None = None,
        attr: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> ET.Element:
        """
        Method to create dictionary from an xml entity.
        """
        element = None
        if tag is not None:
            if overwrite is True:
                for child in parent:
                    if child.tag == tag:
                        element = child
            if element is None:
                element = ET.SubElement(parent, tag)
            if text is not None:
                element.text = str(text)
            if attr is not None:
                for key, val in attr.items():
                    element.set(key, str(val))
        return element

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
