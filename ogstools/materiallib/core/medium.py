import logging
from typing import Any

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .material import Material
from .phase import Phase
from .property import Property

logger = logging.getLogger(__name__)


class Medium:
    """
    A Medium represents a full material definition for a given material_id,
    including its medium-level properties and all relevant phases.

    It constructs its internal structure based on the selected process schema,
    automatically generating phases and components as required.

    Parameters
    ----------
    material_id : int
        The material ID assigned to this medium.
    material : Material
        The solid/medium material to use as property source.
    fluids : dict[str, Material] | None
        A mapping from phase type to fluid material (e.g. {"AqueousLiquid": water, "Gas": co2}).
    process : str
        The process type, e.g. "TH2M_PT". Used to determine required structure.
    """

    def __init__(
        self,
        material_id: int,
        material: Material,
        name: str = "NO_NAME_GIVEN",
        fluids: dict[str, Material] | None = None,
        process: str = "",
    ):
        self.material_id = material_id
        self.material = material
        self.name = name
        self.fluids = fluids or {}
        self.process = process
        self.schema: dict[str, Any] = PROCESS_SCHEMAS.get(process, {})

        if not self.schema:
            msg = f"No schema found for process '{process}'."
            raise ValueError(msg)

        # Initialize phase slots
        self.solid: Phase | None = None
        self.gas: Phase | None = None
        self.aqueous: Phase | None = None
        self.nonaqueous: Phase | None = None
        self.properties: list[Property] = []

        self._build_phases()
        self._load_medium_properties()

    def _build_phases(self) -> None:
        phase_defs = self.schema.get("phases", [])

        for phase_def in phase_defs:
            phase_type = phase_def.get("type")

            match phase_type:
                case "Solid":
                    phase = Phase(
                        phase_type="Solid",
                        solid_material=self.material,
                        process=self.process,
                    )

                case "AqueousLiquid" | "Gas":
                    if phase_type not in self.fluids:
                        msg = (
                            f"Missing fluid material for phase '{phase_type}'."
                        )
                        raise ValueError(msg)

                    phase = Phase(
                        phase_type=phase_type,
                        gas_material=self.fluids.get("Gas"),
                        liquid_material=self.fluids.get("AqueousLiquid"),
                        process=self.process,
                    )

                case _:
                    msg = f"Unsupported phase type: {phase_type}"
                    raise ValueError(msg)

            self.set_phase(phase)

    def _load_medium_properties(self) -> None:
        logger.debug(
            "Loading medium-level properties for material ID %s",
            self.material_id,
        )
        required = set(self.schema.get("properties", []))
        logger.debug("Required medium properties: %s", required)

        self.properties = [
            prop
            for prop in self.material.get_properties()
            if prop.name in required
        ]

        loaded = {prop.name for prop in self.properties}
        missing = required - loaded

        if missing:
            msg = f"Missing required medium properties in material '{self.material.name}': {missing}"
            raise ValueError(msg)

        logger.debug("Loaded %s medium-level properties", len(self.properties))
        logger.debug(self.properties)

    def set_phase(self, phase: Phase) -> None:
        """Assigns a Phase to the correct slot based on its type."""
        match phase.type:
            case "Solid":
                self.solid = phase
            case "Gas":
                self.gas = phase
            case "AqueousLiquid":
                self.aqueous = phase
            case "NonAqueousLiquid":
                self.nonaqueous = phase
            case other:
                msg = f"Unknown phase type: {other}"
                raise ValueError(msg)

    def get_phase(self, phase_type: str) -> Phase | None:
        match phase_type:
            case "Solid":
                return self.solid
            case "Gas":
                return self.gas
            case "AqueousLiquid":
                return self.aqueous
            case "NonAqueousLiquid":
                return self.nonaqueous
            case _:
                return None

    def get_phases(self) -> list[Phase]:
        """Returns all defined phases in order."""
        return [
            p
            for p in [self.solid, self.aqueous, self.nonaqueous, self.gas]
            if p
        ]

    def add_property(self, prop: Property) -> None:
        self.properties.append(prop)

    def __repr__(self) -> str:
        lines = [f"<Medium '{self.name}' (ID={self.material_id})>"]

        # Medium-level properties
        if self.properties:
            lines.append(f"  ├─ {len(self.properties)} medium properties:")
            for prop in self.properties:
                lines.append(f"  │   • {prop.name} ({prop.type})")
        else:
            lines.append("  ├─ no medium-level properties")

        # Phases
        phases = self.get_phases()
        if phases:
            lines.append(f"  ├─ {len(phases)} phase(s):")
            for phase in phases:
                lines.append(
                    f"  │   └─ Phase '{phase.type}' with {len(phase.properties)} properties and {len(phase.components)} components"
                )
                for prop in phase.properties:
                    lines.append(f"  │       • {prop.name} ({prop.type})")
                for comp in phase.components:
                    lines.append(
                        f"  │       └─ Component '{comp.name}' with {len(comp.properties)} properties"
                    )
                    for prop in comp.properties:
                        lines.append(
                            f"  │           • {prop.name} ({prop.type})"
                        )
        else:
            lines.append("  └─ no phases defined")

        return "\n".join(lines)
