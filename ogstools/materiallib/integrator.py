# ogstools/materiallib/integrator.py

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

class MaterialIntegrator:
    def __init__(self, project, process: str = "TH2M"):
        self.prj = project
        self.process = process
        self.schema = PROCESS_SCHEMAS.get(process)
        if self.schema is None:
            raise ValueError(f"Process '{process}' is not defined in PROCESS_SCHEMAS.")

    def add_medium(self, medium):
        for prop in medium.properties:
            self._add_property(medium, prop)

    def _add_property(self, medium, prop):
        # Neues Schema ist flach: { "property_name": "medium" / "solid" }
        allowed = self.schema  # z. B. PROCESS_SCHEMAS["TH2M"]

        if prop.name not in allowed:
            print(f"⚠️  Skipping property '{prop.name}' (not used by {self.process})")
            return

        location = allowed[prop.name]
        args = {
            "medium_id": str(medium.id),
            "name": prop.name,
            "type": prop.type,
            **prop.extra
        }
        if prop.value is not None:
            args["value"] = prop.value
        if location == "solid":
            args["phase_type"] = "Solid"

        self.prj.media.add_property(**args)