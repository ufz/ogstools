# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from pathlib import Path

import numpy as np
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

    def get_curve(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return coords and values of a named curve as numpy arrays.

        For inline curves the space-separated text is parsed directly.
        For file-based curves (read_from_file=true) the binary files are read
        as little-endian float64.

        :param name: Curve name as defined in the <name> element.
        :raises KeyError: If no curve with the given name exists.
        """
        for i, curve in enumerate(self.tree.findall("./curves/curve"), start=1):
            if curve.findtext("name") != name:
                continue
            from_file = (
                curve.findtext("read_from_file") or ""
            ).strip().lower() == "true"
            if from_file:
                # Locate the matching ReferencedFile objects for this curve
                coords_xpath = f"./curves/curve[{i}]/coords"
                values_xpath = f"./curves/curve[{i}]/values"
                rf_coords = next(
                    (rf for rf in self.files if rf._xpath == coords_xpath),
                    None,
                )
                rf_values = next(
                    (rf for rf in self.files if rf._xpath == values_xpath),
                    None,
                )
                if rf_coords is None or rf_values is None:
                    msg = f"Binary files for curve {name!r} are not resolved."
                    raise FileNotFoundError(msg)
                coords = np.fromfile(rf_coords.active_target, dtype="<f8")
                values = np.fromfile(rf_values.active_target, dtype="<f8")
            else:
                coords = np.fromstring(
                    curve.findtext("coords") or "", dtype=float, sep=" "
                )
                values = np.fromstring(
                    curve.findtext("values") or "", dtype=float, sep=" "
                )
            return coords, values
        raise KeyError(name)

    def add_curve(
        self, name: str, coords: Sequence[float], values: Sequence[float]
    ) -> None:
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
