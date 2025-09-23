# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ogstools.ogs6py.project import Project


import logging

from ogstools.materiallib.core.component import Component
from ogstools.materiallib.core.media import MediaSet
from ogstools.materiallib.core.medium import Medium
from ogstools.materiallib.core.phase import Phase
from ogstools.materiallib.core.property import MaterialProperty
from ogstools.ogs6py.media import Media as OGS6pyMedia

logger = logging.getLogger(__name__)


class _ProjectMediaImporter:
    """
    Helper class to import MediaSet data into an OGS6py Project.

    This class encapsulates all logic for translating the materiallib
    hierarchy (MediaSet → Medium → Phase → Component → Property) into
    the XML tree of an OGS project.
    """

    def __init__(self, project: Project):
        self.project = project

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------
    def set_media(self, media_set: MediaSet) -> None:
        """
        Import a complete MediaSet into the Project.

        Parameters
        ----------
        media_set : MediaSet
            Collection of Medium objects derived from a MaterialManager.
        """
        logger.info("Importing %d media into Project.", len(media_set))

        self.project.remove_element("./media")
        self.project.add_block("media", parent_xpath=".")

        self.project.media = OGS6pyMedia(self.project.tree)

        for medium in media_set:
            self.set_medium(medium)

    # ------------------------------------------------------------------
    # Medium
    # ------------------------------------------------------------------
    def set_medium(self, medium: Medium) -> None:
        """Import a single Medium into the Project."""
        mid = str(medium.material_id)
        logger.debug("Importing Medium '%s' (ID=%s)", medium.name, mid)

        for prop in medium.properties:
            self.set_property(prop, medium_id=mid)

        for phase in [
            medium.solid,
            medium.aqueous,
            medium.gas,
            medium.nonaqueous,
        ]:
            if phase:
                self.set_phase(phase, medium_id=mid)

    # ------------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------------
    def set_phase(self, phase: Phase, medium_id: str) -> None:
        """Import a Phase (and its components/properties) into the Project."""
        logger.debug(
            "Importing Phase '%s' (Medium ID=%s)", phase.type, medium_id
        )

        for prop in phase.properties:
            self.set_property(prop, medium_id=medium_id, phase_type=phase.type)

        for comp in phase.components:
            self.set_component(comp, medium_id=medium_id, phase_type=phase.type)

    # ------------------------------------------------------------------
    # Component
    # ------------------------------------------------------------------
    def set_component(
        self, comp: Component, medium_id: str, phase_type: str
    ) -> None:
        """Import a Component (and its properties) into the Project."""
        logger.debug(
            "Importing Component '%s' (Phase=%s, Medium ID=%s)",
            comp.name,
            phase_type,
            medium_id,
        )

        for prop in comp.properties:
            self.set_property(
                prop,
                medium_id=medium_id,
                phase_type=phase_type,
                component_name=comp.name,
            )

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------
    def set_property(
        self,
        prop: MaterialProperty,
        medium_id: str,
        phase_type: str | None = None,
        component_name: str | None = None,
    ) -> None:
        """Import a single MaterialProperty into the Project."""
        args = {
            "medium_id": medium_id,
            "name": prop.name,
            "type": prop.type,
            "value": prop.value,
            **prop.extra,
        }
        if phase_type is not None:
            args["phase_type"] = phase_type
        if component_name is not None:
            args["component_name"] = component_name

        logger.debug(
            "Adding property '%s' (type=%s, medium_id=%s, phase=%s, component=%s)",
            prop.name,
            prop.type,
            medium_id,
            phase_type,
            component_name,
        )

        self.project.media.add_property(**args)
