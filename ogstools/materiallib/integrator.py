# ogstools/materiallib/integrator.py

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS
from ogstools.materiallib.core.core import Property

class MaterialIntegrator:
    def __init__(self, project, process: str = "TH2M"):
        self.prj = project
        self.process = process
        self.schema = PROCESS_SCHEMAS.get(process)
        if self.schema is None:
            raise ValueError(f"Process '{process}' is not defined in PROCESS_SCHEMAS.")

    def add_medium(self, medium):
        for phase_type, props in medium.phases.items():
            for prop in props:
                self._add_property(medium, prop, phase_type=phase_type)

        # for phase_type, comps in medium.components.items():
        #     for comp_name, comp_props in comps.items():
        #         for prop in comp_props:
        #             self._add_property(medium, prop, phase_type=phase_type, component_name=comp_name)



    def _add_property(self, medium, prop, phase_type: str, component_name: str = None):
        args = {
            "medium_id": str(medium.id),
            "name": prop.name,
            "type": prop.type,
            **prop.extra
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
