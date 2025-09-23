# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging
from collections.abc import Iterator

from ogstools.materiallib.core.material_manager import MaterialManager
from ogstools.materiallib.core.medium import Medium

logger = logging.getLogger(__name__)


class MediaSet:
    """
    Represents a collection of Medium objects (solids or fluids) for a given process.

    MediaSet are constructed from a filtered MaterialManager, i.e. after process
    schemas, subdomain assignments, and fluid materials have already been applied.

    Provides:
    - Dictionary-like access (`__getitem__`, `keys()`, `values()`, `items()`).
    - Iteration over all Medium instances.
    - Lookup by name or by material ID.

    Notes
    -----
    This class requires that the input MaterialManager has already been
    filtered for a specific process (``filtered_db.process`` must not be None).
    """

    def __init__(self, filtered_db: MaterialManager):
        """
        Create a MediaSet collection from a filtered MaterialManager.

        Parameters
        ----------
        filtered_db : MaterialManager
            A MaterialManager instance that has been filtered for a
            specific process and contains subdomain IDs and fluids.

        Raises
        ------
        ValueError
            If `filtered_db.process` is None (i.e. unfiltered manager),
            or if any Medium fails validation.
        """
        if filtered_db.process is None:
            logger.error(
                "Cannot construct MediaSet: MaterialManager has no process."
            )
            msg = (
                "MediaSet can only be created from a filtered MaterialManager."
            )
            raise ValueError(msg)

        self.process = filtered_db.process
        self._media: list[Medium] = []
        self._name_map: dict[str, Medium] = {}

        for name, mat_id in filtered_db.subdomain_ids.items():
            mat = filtered_db.materials_db[name]
            medium = Medium(
                material_id=mat_id,
                material=mat,
                name=name,
                fluids=filtered_db.fluids,
                process=self.process,
            )
            logger.debug("Created Medium '%s' (ID=%d)", name, mat_id)

            self._media.append(medium)
            self._name_map[name] = medium

            if not self.validate_medium(medium):
                msg = f"Medium '{name}' (ID={mat_id}) is invalid."
                logger.error(msg)
                raise ValueError(msg)

    def __iter__(self) -> Iterator[Medium]:
        """Iterate over all Medium objects."""
        return iter(self._media)

    def __getitem__(self, key: str) -> Medium:
        """Retrieve a Medium by its subdomain name."""
        return self._name_map[key]

    def __len__(self) -> int:
        """Return the number of Medium objects."""
        return len(self._media)

    def __contains__(self, name: str) -> bool:
        """Return True if a Medium with the given name exists."""
        return name in self._name_map

    def keys(self) -> list[str]:
        """Return the list of subdomain names (keys)."""
        return list(self._name_map.keys())

    def values(self) -> list[Medium]:
        """Return the list of Medium objects (values)."""
        return list(self._name_map.values())

    def items(self) -> list[tuple[str, Medium]]:
        """Return (name, Medium) pairs like dict.items()."""
        return list(self._name_map.items())

    def to_dict(self) -> dict[str, Medium]:
        """Return the mapping of subdomain names to Medium objects."""
        return dict(self._name_map)

    def get_by_id(self, material_id: int) -> Medium | None:
        """
        Lookup a Medium by its material ID.

        Parameters
        ----------
        material_id : int
            The material_id assigned to a subdomain.

        Returns
        -------
        Medium | None
            The Medium object with the given ID, or None if not found.
        """
        return next(
            (m for m in self._media if m.material_id == material_id), None
        )

    @classmethod
    def from_project(cls, prj: str, process: str) -> "MediaSet":
        """
        Reconstruct a Media collection from an OGS6py Project.

        Parameters
        ----------
        prj : Project
            An OGS6py Project instance containing <media> definitions.
        process : str
            The process type to which these media belong.

        Raises
        ------
        NotImplementedError
            This functionality is not implemented yet.
        """
        logger.info("Attempted to parse Media from Project (not implemented).")
        _ = prj, process  # prevent unused arg warnings
        msg = "from_prj() not implemented yet."
        raise NotImplementedError(msg)

    def validate(self) -> bool:
        """
        Validate all Medium objects.

        Returns
        -------
        bool
            True if all Medium objects are valid, otherwise raises ValueError.
        """
        logger.debug("Validating %d media objects...", len(self._media))
        for medium in self._media:
            medium.validate()
        return True

    def validate_medium(self, medium: Medium) -> bool:
        """
        Validate a single Medium.

        Parameters
        ----------
        medium : Medium
            The Medium object to validate.

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        ValueError
            If the Medium fails validation.
        """
        if not medium.validate():
            msg = f"Medium '{medium.name}' failed validation."
            logger.error(msg)
            raise ValueError(msg)
        return True

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        """Return a human-readable string representation of this MediaSet collection."""
        lines = [f"<MediaSet with {len(self)} entries>"]
        for medium in self._media:
            for line in repr(medium).splitlines():
                lines.append("  " + line)
        return "\n".join(lines)
