# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import shutil
from pathlib import Path

from lxml import etree as ET

from ogstools.core.storage import StorageBase
from ogstools.ogs6py import build_tree


class ReferencedFile(build_tree.BuildTree, StorageBase):
    """
    Base class for a single file referenced in an OGS project file.

    Subclasses define the XML location of the filename via class attributes:
    - _NAME: StorageBase name
    - _EXT:  file extension (without dot)
    - _XPATH: XPath expression locating the filename element in the tree

    Alternatively, pass xpath= to __init__ to override the class-level _XPATH
    at the instance level (used for dynamically located references such as
    curve binary files).
    """

    __hash__ = None
    _NAME: str = ""
    _EXT: str = ""
    _XPATH: str = ""

    def __init__(self, tree: ET.ElementTree, xpath: str | None = None) -> None:
        build_tree.BuildTree.__init__(self, tree)
        StorageBase.__init__(self, self._NAME or "ReferencedFile", self._EXT)
        self.root = self.tree.getroot()
        self._xpath = xpath if xpath is not None else self._XPATH

    @property
    def filename(self) -> str | None:
        """Get the filename from the XML tree."""
        elem = self.root.find(self._xpath)
        if elem is not None and elem.text:
            return elem.text.strip() or None
        return None

    def _propagate_target(self) -> None:
        """No children to propagate to."""

    def _save_impl(self, dry_run: bool = False) -> list[Path]:
        if not self.filename:
            return []

        target = self.next_target

        if dry_run:
            return [target]

        target.parent.mkdir(parents=True, exist_ok=True)

        if (
            self.active_target
            and self.active_target.exists()
            and self.active_target.resolve() != target.resolve()
        ):
            shutil.copy2(self.active_target, target)

        return [target]

    def save(
        self,
        target: Path | str | None = None,
        overwrite: bool | None = None,
        dry_run: bool = False,
        archive: bool = False,
        id: str | None = None,
    ) -> list[Path]:
        if not self.filename:
            return []

        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run)
        if files:
            self._post_save(user_defined, archive, dry_run)
        return files

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReferencedFile):
            return NotImplemented
        return self.filename == other.filename

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filename={self.filename!r}, is_saved={self.is_saved})"

    def __str__(self) -> str:
        if not self.filename:
            return f"{self.__class__.__name__}: (no file defined)"
        return f"{self.__class__.__name__}: {self.filename}"
