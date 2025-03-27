# ogstools/materiallib/integrator.py

from typing import Any

from ogs6py import Project

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .core.medium import Medium


class MaterialIntegrator:
    def __init__(self, project: Project, process: str = "TH2M"):
        self.prj = project
        self.process = process
        self.schema = PROCESS_SCHEMAS.get(process)
        if self.schema is None:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)

    def add_medium(self, medium: Medium) -> None:
        pass
        # for phase_type, props in medium.phases.items():
        #     for prop in props:
        #         self._add_property(medium, prop, phase_type=phase_type)

        # for phase_type, comps in medium.components.items():
        #     for comp_name, comp_props in comps.items():
        #         for prop in comp_props:
        #             self._add_property(medium, prop, phase_type=phase_type, component_name=comp_name)

    def _add_property(
        self,
        # medium: Medium,
        prop: Any,
        phase_type: str,
        component_name: str | None = None,
    ) -> None:
        args = {
            # "medium_id": str(medium.id),
            "name": prop.name,
            "type": prop.type,
            **prop.extra,
        }
        if prop.value is not None:
            args["value"] = prop.value
        args["phase_type"] = phase_type
        if component_name:
            args["component_name"] = component_name

        self.prj.media.add_property(**args)

    # def _add_property(self, medium, prop, phase_type=None, component_name=None):
    #     location = self.schema.get(prop.name, "medium")
    #     args = {
    #         "medium_id": str(medium.id),
    #         "name": prop.name,
    #         "type": prop.type,
    #         **prop.extra
    #     }
    #     if prop.value is not None:
    #         args["value"] = prop.value
    #     if location == "solid":
    #         args["phase_type"] = "Solid"
    #     if phase_type:
    #         args["phase_type"] = phase_type
    #     if component_name:
    #         args["component_name"] = component_name

    #     self.prj.media.add_property(**args)
