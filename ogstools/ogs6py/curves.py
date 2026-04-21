# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from lxml import etree as ET

from ogstools.ogs6py import build_tree


class Curves(build_tree.BuildTree):
    """
    Class to create the curve section of the project file.
    """

    def __init__(
        self, tree: ET.ElementTree, input_file: Path | None = None
    ) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.curves = self.populate_tree(self.root, "curves", overwrite=True)
        self._input_file = input_file
        self.files: list = []
        self._reload_curve_files()

    def _reload_curve_files(self) -> None:
        """Rebuild curve_files from current XML tree (curves with read_from_file=true)."""
        from ogstools.ogs6py import referenced_file as referenced_file_module

        self.files = []
        for i, curve in enumerate(self.tree.findall("./curves/curve"), start=1):
            rfb = curve.find("read_from_file")
            if rfb is None or (rfb.text or "").strip().lower() != "true":
                continue
            for tag in ("coords", "values"):
                elem = curve.find(tag)
                if elem is not None and elem.text and elem.text.strip():
                    xpath = f"./curves/curve[{i}]/{tag}"
                    rf = referenced_file_module.ReferencedFile(
                        self.tree, xpath=xpath
                    )
                    if self._input_file is not None and rf.filename:
                        src = self._input_file.parent / rf.filename
                        if src.exists():
                            rf._active_target = src
                    self.files.append(rf)

    def add_curve(self, name: str, coords: list, values: list) -> None:
        """
        Adds a new curve.

        :param name:
        :param coords:
        :param values:
        """
        if len(coords) != len(values):
            msg = """Number of time coordinate points differs \
                     from number of values"""
            raise ValueError(msg)
        curve = self.populate_tree(self.curves, "curve")
        self.populate_tree(curve, "name", name)
        coord_str = ""
        value_str = ""
        for i, coord in enumerate(coords):
            if i < (len(coords) - 1):
                coord_str = coord_str + str(coord) + " "
                value_str = value_str + str(values[i]) + " "
            if i == (len(coords) - 1):
                coord_str = coord_str + str(coord)
                value_str = value_str + str(values[i])
        self.populate_tree(curve, "coords", text=coord_str)
        self.populate_tree(curve, "values", text=value_str)
