from typing import Any

from .phase import Phase


class Medium:
    """
    A Medium is a hierarchical structure of physical properties used in
    an OGS simulation to describe a material group (material_id).

    A Medium can contain up to one phase of each of the following types:
    - Solid
    - Gas
    - AqueousLiquid
    - NonAqueousLiquid

    Each phase holds its own set of properties and may contain components
    (e.g. "water", "carbon_dioxide"), which can also have their own properties.

    Medium-level properties (e.g. porosity, permeability) are stored in
    the `properties` attribute of the Medium itself.

    Phases can be assigned using the `set_phase()` method, which places the
    given Phase in the correct internal slot based on its `type`.

    Use `get_phases()` to retrieve a list of all assigned phases in canonical order.
    """

    def __init__(
        self,
        material_id: int,
        properties: Any = None,
        solid: Phase | None = None,
        gas: Phase | None = None,
        aqueous: Phase | None = None,
        nonaqueous: Phase | None = None,
    ):
        self.material_id = material_id
        self.properties = properties or []

        # specific phases - max. one per type
        self.solid = solid
        self.gas = gas
        self.aqueous = aqueous
        self.nonaqueous = nonaqueous

    def add_property(self, prop: Any) -> None:
        self.properties.append(prop)

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

    def get_phases(self) -> list[Phase]:
        """Returns all defined phases in order."""
        return [
            p
            for p in [self.solid, self.aqueous, self.nonaqueous, self.gas]
            if p
        ]

    def __repr__(self) -> str:
        lines = [f"<Medium ID={self.material_id}>"]

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
                for comp in phase.components:
                    lines.append(
                        f"  │       └─ Component '{comp.name}' with {len(comp.properties)} properties"
                    )
        else:
            lines.append("  └─ no phases defined")

        return "\n".join(lines)
