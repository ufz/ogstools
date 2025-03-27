from pathlib import Path
import yaml
from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS
from ogstools.definitions import MATERIALS_DIR


# class Medium:
#     """
#     Ziel:
#         Medium
#             Gasphase
#                 Komponente 1
#                     Eigenschaften (K1)
#                 Komponente 2
#                     Eigenschaften (K2)
#                 Eigenschaften (G)
#             Flüssigphase
#                 Komponente 1
#                     Eigenschaften (K1)
#                 Komponente 2
#                     Eigenschaften (K2)
#                 Eigenschaften (L)
#             Feste phase
#                 Eigenschaften (S)
#             Eigenschaften (M)

#     Eingabe:    MaterialList-Objekt
#                 Process_Schemas (für zuordnung eigenschaft -> Phase od. Medium od. Komponente)

#     """
#     def __init__(self, materiallist):
#         self.schema = PROCESS_SCHEMAS.get(self.process)

#         """
#         return medium-objekt mit gas- Wasser, Festkörperphase, etc.

#         """

#     def to_ogs_properties(self) -> list[dict]:
#         return [prop.to_ogs_dict() for prop in self.properties]

#     def add_phase(self, phase_type: str, properties: list, components: dict[str, list] = None):
#         self.phases[phase_type] = properties
#         if components:
#             self.components[phase_type] = components

#     def add_to_project(self, prj):
#         from ogstools.materiallib.integrator import MaterialIntegrator
#         MaterialIntegrator(prj, process=self.process).add_medium(self)


class MaterialLib:
    def __init__(self, data_dir: Path = None, process: str = "TH2M"):
        self.materials = {}
        self.data_dir = data_dir or Path(__file__).parents[1] / "data"
        self.process = process  # save OGS-process
        # self._load_materials()

    # def _load_materials(self):
    #     yaml_files = list(self.data_dir.glob("*.yml")) + list(self.data_dir.glob("*.yaml"))
    #     if not yaml_files:
    #         raise FileNotFoundError(f"No YAML files found in {self.data_dir}")
    #     for file_path in yaml_files:
    #         with open(file_path, "r", encoding="utf-8") as file:
    #             data = yaml.safe_load(file)
    #             name = data.get("name", file_path.stem)
    #             self.materials[name] = data

    def create_medium(
        self,
        name: str,
        id_: int,
        overrides: dict = None,
        fluids: dict[str, str] = None,
    ) -> Medium:
        data = self.get_material(name)
        if data is None:
            raise ValueError(f"Material '{name}' not found in database.")
        if overrides:
            data["properties"] = {**data.get("properties", {}), **overrides}

        schema = PROCESS_SCHEMAS.get(self.process)
        if not schema:
            raise ValueError(f"No schema defined for process '{self.process}'")

        solid_props, medium_props = [], []
        # medium = Medium(id_=id_, name=name, properties=[], process=self.process)

        for pname, plist in data.get("properties", {}).items():
            print(type(plist))
            print(plist)
            print(pname)
            entry = plist[0]
            prop = Property(
                name=pname,
                type_=entry.get("type", "Constant"),
                value=entry.get("value"),
                **{
                    k: v for k, v in entry.items() if k not in ["type", "value"]
                },
            )
            location = schema.get(pname)
            if location == "solid":
                solid_props.append(prop)
            elif location == "medium":
                medium_props.append(prop)
            else:
                print(
                    f"⚠️  Property '{pname}' not recognized in schema – ignored."
                )

        # medium.add_phase("Solid", solid_props)
        # medium.add_phase("Medium", medium_props)

        # 🔽 Fluide ergänzen (optional)
        if fluids:
            fluidschema = schema.get("_fluids", {})
            use_components = "component_properties" in next(
                iter(fluidschema.values()), {}
            )

            for phase_type, fluid_name in fluids.items():
                fdata = self.get_material(fluid_name)
                if not fdata:
                    raise ValueError(f"Fluid '{fluid_name}' not found in DB.")
                all_props = fdata.get("properties", {})
                if not all_props:
                    raise ValueError(f"No properties in fluid '{fluid_name}'.")

                props_phase, props_comp = [], []
                required = fluidschema.get(phase_type, {})

                for pname, plist in all_props.items():
                    entry = plist[0]
                    prop = Property(
                        name=pname,
                        type_=entry.get("type", "Constant"),
                        value=entry.get("value"),
                        **{
                            k: v
                            for k, v in entry.items()
                            if k not in ["type", "value"]
                        },
                    )

                    if use_components and pname in required.get(
                        "component_properties", []
                    ):
                        props_comp.append(prop)
                    elif pname in required.get("phase_properties", []):
                        props_phase.append(prop)

                if use_components:
                    pass
                    # medium.add_phase(phase_type, props_phase, components={fluid_name: props_comp})
                else:
                    pass
                    # medium.add_phase(phase_type, props_phase)

        return  # medium
