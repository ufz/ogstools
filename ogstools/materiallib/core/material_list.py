from typing import Any, cast

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
    [5] matID="X", name="water", properties: ["Density", "Viscosity", ...]
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

        self.materials: dict[str, Material] = (
            {}
        )  # e.g. {"host_rock": <Material "opalinus_clay">}
        self.material_ids: dict[str, int] = {}  # e.g. {"host_rock": 2}

        self.fluid_materials: dict[str, Material] = (
            {}
        )  # {"AqueousLiquid": Material(...)}

        self._build_material_list()

    def _build_material_list(self) -> None:
        # Solid materials (subdomain-based)
        for entry in self.subdomains:
            name = entry["material"]
            ids = entry["material_id"]
            subdomain_names = entry.get("subdomain", [])
            raw = self.material_db.get_material(name)

            if raw is None:
                msg = f"Material '{name}' not found in database."
                raise ValueError(msg)

            required_names = self.get_required_property_names()

            filtered_props = [
                p for p in raw.get_properties() if p.name in required_names
            ]
            material = Material(name=name, properties=filtered_props)

            for subdomain_name, mat_id in zip(
                subdomain_names, ids, strict=False
            ):
                if subdomain_name not in self.materials:
                    self.materials[subdomain_name] = material
                    self.material_ids[subdomain_name] = mat_id

        # Fluid materials (they have no material_id)
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

    # def get_required_property_names(self) -> set[str]:
    #     """
    #     Returns a set of all property names required by the current process schema.

    #     Prints a structured overview of the required properties, grouped by hierarchy level):
    #     - Medium-level properties
    #     - Phase-level properties
    #     - Component-level properties (if fluid)

    #     """
    #     if self.schema is None:
    #         raise ValueError("Process schema not set. Cannot determine required properties.")

    #     required = set()

    #     # Print an overview of the required property structure
    #     print("=== Required Property Structure ===")
    #     print("===-----------------------------===")

    #     # Medium-level properties
    #     medium_props = self.schema.get("properties", [])
    #     if medium_props:
    #         print("Medium-level properties:")
    #         for prop in medium_props:
    #             print(f"  - {prop}")
    #         print()
    #         required.update(medium_props)

    #     # Phases and their properties
    #     for phase in self.schema.get("phases", []):
    #         phase_type = phase.get("type")

    #         # Phase-level
    #         phase_props = phase.get("properties", [])
    #         if phase_props:
    #             print(f"Phase-level properties for '{phase_type}':")
    #             for prop in phase_props:
    #                 print(f"  - {prop}")
    #             print()
    #             required.update(phase_props)

    #         # Component-level (if fluid)
    #         if phase_type in ["AqueousLiquid", "Gas"]:
    #             components = phase.get("components", {})
    #             for comp_name, comp_props in components.items():
    #                 if comp_props:
    #                     print(f"Component-level properties for '{comp_name}' in phase '{phase_type}':")
    #                     for prop in comp_props:
    #                         print(f"  - {prop}")
    #                     print()
    #                     required.update(comp_props)
    #     print("===-----------------------------===")

    #     return required

    def get_required_property_names(self) -> set[str]:
        """
        Returns a set of all property names required by the current process schema.
        This includes medium-level, phase-level, and component-level properties.
        """
        if self.schema is None:
            msg = (
                "Process schema not set. Cannot determine required properties."
            )
            raise ValueError(msg)

        required = set[str]()
        PHASES_WITH_COMPONENTS = {"AqueousLiquid", "Gas", "NonAqueousLiquid"}

        # Medium-level
        medium_properties = cast(list[str], self.schema.get("properties", []))
        required.update(medium_properties)

        # Phase-level and component-level
        phases = cast(list[dict[str, Any]], self.schema.get("phases", []))
        for phase in phases:
            required.update(phase.get("properties", []))
            if phase.get("type") in PHASES_WITH_COMPONENTS:
                for component_props in phase.get("components", {}).values():
                    required.update(component_props)

        return required

    def get_material(self, name: str) -> Material | None:
        return self.materials.get(name)

    def list_ids(self) -> list[int]:
        return list(self.material_ids.values())

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
