# ogstools/materiallib/integrator.py

from typing import Any

from ogs6py import Project

from ogstools.materiallib.core.medium import Medium
from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS


class MaterialIntegrator:
    def __init__(self, project: Project, process: str = "TH2M"):
        self.prj = project
        self.process = process
        self.schema = PROCESS_SCHEMAS.get(process)
        if self.schema is None:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)

    def add_media(self, media: list[Medium]) -> None:
        for medium in media:
            self.add_medium(medium)

    def add_medium(self, medium: Medium) -> None:
        mid = str(medium.material_id)
        self.prj.add_block("medium", {"id": mid}, parent_xpath="./media")

        # Medium-level properties
        for prop in medium.properties:
            self._add_property(prop, medium_id=mid)

        # Phases
        for phase in medium.get_phases():
            self.prj.add_block(
                "phase", parent_xpath=f"./media/medium[@id='{mid}']/phases"
            )
            self.prj.add_element(
                parent_xpath=f"./media/medium[@id='{mid}']/phases/phase[last()]",
                tag="type",
                text=phase.type,
            )

            # Phase-level properties
            for prop in phase.properties:
                self._add_property(
                    prop,
                    medium_id=mid,
                    phase_type=phase.type,
                )

            # Components
            for comp in phase.components:
                self.prj.add_block(
                    "component",
                    parent_xpath=f"./media/medium[@id='{mid}']/phases/phase[last()]/components",
                )
                self.prj.add_element(
                    parent_xpath=f"./media/medium[@id='{mid}']/phases/phase[last()]/components/component[last()]",
                    tag="name",
                    text=comp.name,
                )

                for prop in comp.properties:
                    self._add_property(
                        prop,
                        medium_id=mid,
                        phase_type=phase.type,
                        component_name=comp.name,
                    )

    def _add_property(
        self,
        prop: Any,
        medium_id: str,
        phase_type: str | None = None,
        component_name: str | None = None,
    ) -> None:
        args = {
            "medium_id": medium_id,
            "name": prop.name,
            "type": prop.type,
            **prop.extra,
        }
        if prop.value is not None:
            args["value"] = prop.value
        if phase_type:
            args["phase_type"] = phase_type
        if component_name:
            args["component_name"] = component_name

        self.prj.media.add_property(**args)
