from typing import cast

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .material import Material
from .material_db import MaterialDB


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

    def __init__(
        self,
        material_db: MaterialDB,
        subdomains: list[dict],
        fluids: dict[str, str] | None = None,
        process: str = "TH2M",
    ):
        self.material_db = material_db
        self.subdomains = subdomains
        self.fluids = fluids or {}
        self.process = process
        self.schema = PROCESS_SCHEMAS.get(process)

        if self.schema is None:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)

        self.materials: dict[int, Material] = {}  # {material_id: Material}
        self.fluid_materials: dict[str, Material] = (
            {}
        )  # {"AqueousLiquid": Material(...)}

        self._build_material_list()

    def _build_material_list(self) -> None:
        for entry in self.subdomains:
            name = entry["material"]
            ids = entry["material_id"]
            raw = self.material_db.get_material(name)

            if raw is None:
                msg = f"Material '{name}' not found in database."
                raise ValueError(msg)

            # For now, we collect all required properties from the process schema, independent of material type
            required_names = self.get_required_property_names()

            # Solid materials
            for mat_id in ids:
                if mat_id in self.materials:
                    continue

                filtered_props = [
                    p for p in raw.get_properties() if p.name in required_names
                ]
                material = Material(name=name, properties=filtered_props)
                # print(f"Solids: {filtered_props}")
                self.materials[mat_id] = material

        # Fluid materials (they have no MaterialID)
        for phase_type, mat_name in self.fluids.items():
            raw = self.material_db.get_material(mat_name)

            if raw is None:
                msg = f"Fluid material '{mat_name}' not found in database."
                raise ValueError(msg)

            filtered_props = [
                p for p in raw.get_properties() if p.name in required_names
            ]
            material = Material(name=mat_name, properties=filtered_props)
            self.fluid_materials[phase_type] = material

    def get_required_property_names(self) -> set[str]:
        """
        Returns a set of all property names required by the current process schema.
        This includes medium, solid, phase and component properties.
        """
        if self.schema is None:
            msg = (
                "Process schema not set. Cannot determine required properties."
            )
            raise ValueError(msg)

        required = set()

        for key, value in self.schema.items():
            if key == "_fluids":
                fluid_defs = cast(dict[str, dict[str, list[str]]], value)
                for fluid_def in fluid_defs.values():
                    required.update(fluid_def.get("phase_properties", []))
                    required.update(fluid_def.get("component_properties", []))
            else:
                required.add(key)

        return required

    def get_material(self, material_id: int) -> Material | None:
        return self.materials.get(material_id)

    def list_ids(self) -> list[int]:
        return list(self.materials.keys())

    def list_materials(self) -> list[str]:
        return [mat.name for mat in self.materials.values()]

    def __repr__(self) -> str:
        lines = [f"<MaterialList for process '{self.process}'>"]

        if self.materials:
            lines.append(
                f"  ├─ {len(self.materials)} solid material entries mapped to material_ids:"
            )
            for mid, mat in sorted(self.materials.items()):
                lines.append(f"  │   [{mid}] → {mat.name}")
        else:
            lines.append("  ├─ No solid or medium materials defined")

        if hasattr(self, "fluid_materials") and self.fluid_materials:
            lines.append(f"  ├─ {len(self.fluid_materials)} fluid materials:")
            for phase_type, mat in self.fluid_materials.items():
                lines.append(f"  │   {phase_type}: {mat.name}")
        else:
            lines.append("  └─ No fluid materials assigned")

        return "\n".join(lines)
