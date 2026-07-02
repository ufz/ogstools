# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from lxml import etree as ET

from ogstools.core.storage import StorageBase
from ogstools.ogs6py import build_tree


class ReferencedFile(build_tree.BuildTree, StorageBase):
    """Represents a single file referenced by path in an OGS project file."""

    __hash__ = None
    # Subclasses set these to fix the XML location of the filename element.
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
        """Filename as stored in the project file, or None if not set."""
        if not self._xpath:
            return None
        elem = self.root.find(self._xpath)
        if elem is not None and elem.text:
            return elem.text.strip() or None
        return None

    @property
    def is_file(self) -> bool:
        return True

    def _propagate_target(self) -> None:
        pass

    def _save_impl(self, dry_run: bool = False) -> list[Path]:
        if not self.filename:
            return []

        target = self.next_target

        if dry_run:
            return [target]

        if self.active_target and self.active_target.exists():
            self.link(target, dry_run=False)

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
