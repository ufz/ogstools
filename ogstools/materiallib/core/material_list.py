from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS
from .material import RawMaterial, Material

class MaterialList:
    """
    Represents a filtered material set, tailored to a specific OGS process
    and a set of subdomains (material_id → material_name mapping).

    This class loads RawMaterial objects from a MaterialDB instance,
    filters them according to the selected process schema,
    and stores validated Material instances ready for use in media or phases.

    Parameters
    ----------
    material_db : MaterialDB
        A database providing unfiltered RawMaterial objects.
    subdomains : list of dicts
        Each dict: {material_id: int, material: str, subdomain: [str]}
    process : str
        The OGS process type to determine required/optional properties.

    Returns
    -------
    A MaterialList instance containing filtered Material objects for each
    material_id defined in the subdomains.

    Notes
    -----
    Only Material instances created by this class should be used in the simulation.
    RawMaterial objects must not be used to build media or phases.

    Example
    -------
    [1] matID="0", name="opa", properties: ["Density", "Permeability", ...]
    [2] matID="1", name="bentonit", properties: ["Density", "Permeability", ...]
    [3] matID="2", name="concrete", properties: ["Density", "Permeability", ...]
    [4] matID="3", name="concrete", properties: ["Density", "Permeability", ...]
    [5] matID="4", name="water", properties: ["Density", "Viscosity", ...]
    """

    def __init__(self, material_db, subdomains: list[dict], fluids: dict[str, str] = None, process: str = "TH2M"):
        self.material_db = material_db
        self.subdomains = subdomains
        self.fluids = fluids or {}
        self.process = process
        self.schema = PROCESS_SCHEMAS.get(process)

        if self.schema is None:
            raise ValueError(f"No process schema found for '{process}'")

        self.materials: dict[int, Material] = {}  # {material_id: Material}
        self.fluid_materials: dict[str, Material] = {}  # {"AqueousLiquid": Material(...)}

        self._build_material_list()

    def _build_material_list(self):
        for entry in self.subdomains:
            name = entry["material"]
            ids = entry["material_id"]
            raw = self.material_db.get_material(name)

            if raw is None:
                raise ValueError(f"Material '{name}' not found in database.")
            if not isinstance(raw, RawMaterial):
                raise TypeError(f"Expected RawMaterial from MaterialDB, got {type(raw)}")

            # For now, we collect all required properties from the process schema, independent of material type
            required_names = self.get_required_property_names()

            # Solid materials
            for mat_id in ids:
                if mat_id in self.materials:
                    continue

                filtered_props = [p for p in raw.get_properties() if p.name in required_names]
                material = Material(name=name, properties=filtered_props)
                # print(f"Solids: {filtered_props}")
                print(f"MaterialID: {mat_id}")
                for p in filtered_props:
                    print("Solids:", p.name, p.type, p.value)
                print("....")
                self.materials[mat_id] = material

        # Fluid materials (they have no MaterialID)
        for phase_type, mat_name in self.fluids.items():
            raw = self.material_db.get_material(mat_name)

            if raw is None:
                raise ValueError(f"Fluid material '{mat_name}' not found in database.")
            if not isinstance(raw, RawMaterial):
                raise TypeError(f"Expected RawMaterial, got {type(raw)}")

            filtered_props = [p for p in raw.get_properties() if p.name in required_names]
            material = Material(name=mat_name, properties=filtered_props)
            # print(f"Fluids: {filtered_props}")
            for p in filtered_props:
                print("Fluids:", p.name, p.type, p.value)
            self.fluid_materials[phase_type] = material
    
    def get_required_property_names(self) -> set[str]:
        """
        Returns a set of all property names required by the current process schema.
        This includes medium, solid, phase and component properties.
        """
        required = set()

        for key, value in self.schema.items():
            if key == "_fluids":
                for fluid_def in value.values():
                    required.update(fluid_def.get("phase_properties", []))
                    required.update(fluid_def.get("component_properties", []))
            else:
                required.add(key)

        return required


    def get_material(self, material_id: int):
        return self.materials.get(material_id)

    def list_ids(self) -> list[int]:
        return list(self.materials.keys())

    def list_materials(self) -> list[str]:
        return [mat.name for mat in self.materials.values()]

    def __repr__(self):
        lines = [
            f"<MaterialList for process '{self.process}'>",
            f"  ├─ {len(self.materials)} materials mapped to material_ids:",
        ]
        for mid, mat in sorted(self.materials.items()):
            lines.append(f"  │   [{mid}] → {mat.name}")
        return "\n".join(lines)
