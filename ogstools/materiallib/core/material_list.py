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

    def __init__(self, material_db, subdomains: list[dict], process: str = "TH2M"):
        self.material_db = material_db
        self.subdomains = subdomains
        self.process = process
        self.schema = PROCESS_SCHEMAS.get(process)

        if self.schema is None:
            raise ValueError(f"No process schema found for '{process}'")

        self.materials: dict[int, Material] = {}  # {material_id: Material}
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

            for mat_id in ids:
                if mat_id in self.materials:
                    continue

                # Here we should filter raw.properties according to the process schema
                # For now, take all properties
                filtered_props = raw.get_properties()  # Replace with actual filtering later
                material = Material(name=name, properties=filtered_props)

                self.materials[mat_id] = material

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
