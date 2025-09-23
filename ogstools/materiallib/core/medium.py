# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging
from typing import Any

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .material import Material
from .phase import Phase
from .property import MaterialProperty

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

        if fluids is not None:
            checked_fluids = {}
            for key, val in fluids.items():
                if not isinstance(val, Material):
                    msg = f"Fluid '{key}' must be a Material, got {type(val).__name__}"  # type: ignore[unreachable]
                    raise TypeError(msg)
                checked_fluids[key] = val
            self.fluids = checked_fluids
        else:
            self.fluids = {}

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
        self.properties: list[MaterialProperty] = []

        self._build_phases()
        self._load_medium_properties()

    def _build_phases(self) -> None:
        """
        Build all phases for this Medium according to the process schema.

        Supported cases:
        1. Single-phase without components
        2. Two-phase without components
        3. Two-phase with components (TH2M-phase transitions)
        Not yet supported:
        4. Single-phase with components (multi-component single-phase transport)
        """
        phase_defs = self.schema.get("phases", [])

        for phase_def in phase_defs:
            phase_type = phase_def.get("type")
            components_def = phase_def.get("components")

            match phase_type:
                # -------------------
                # SOLID PHASE
                # -------------------
                case "Solid":
                    phase = Phase(
                        phase_type="Solid",
                        solid_material=self.material,
                        process=self.process,
                    )

                # -------------------
                # FLUID PHASES (Gas / AqueousLiquid)
                # -------------------
                case "Gas" | "AqueousLiquid":
                    if components_def:
                        # --- Case 3: Two-phase with components ---
                        other_phase = (
                            "AqueousLiquid" if phase_type == "Gas" else "Gas"
                        )
                        all_phase_types = [p.get("type") for p in phase_defs]

                        if other_phase not in all_phase_types:
                            # --- Case 4: Single-phase with components ---
                            msg = f"Single-phase multi-component case for {phase_type} is not yet implemented."
                            raise NotImplementedError(msg)

                        # Require both fluids
                        if not (
                            self.fluids.get("Gas")
                            and self.fluids.get("AqueousLiquid")
                        ):
                            msg = f"Both Gas and AqueousLiquid fluids required for {phase_type} with components."
                            raise ValueError(msg)

                        phase = Phase(
                            phase_type=phase_type,
                            gas_material=self.fluids.get("Gas"),
                            liquid_material=self.fluids.get("AqueousLiquid"),
                            process=self.process,
                        )

                    else:
                        # --- Case 1 + 2: No components ---
                        if phase_type == "Gas":
                            if not self.fluids.get("Gas"):
                                msg = "Gas phase requires gas_material."
                                raise ValueError(msg)
                            phase = Phase(
                                phase_type="Gas",
                                gas_material=self.fluids["Gas"],
                                process=self.process,
                            )

                        elif phase_type == "AqueousLiquid":
                            if not self.fluids.get("AqueousLiquid"):
                                msg = "AqueousLiquid phase requires liquid_material."
                                raise ValueError(msg)
                            phase = Phase(
                                phase_type="AqueousLiquid",
                                liquid_material=self.fluids["AqueousLiquid"],
                                process=self.process,
                            )

                # -------------------
                # UNSUPPORTED
                # -------------------
                case _:
                    msg = f"Unsupported phase type: {phase_type}"
                    raise ValueError(msg)

            # Validate and register the phase
            if not self.validate_phase(phase):
                msg = (
                    f"Phase '{phase_type}' in medium '{self.name}' is invalid."
                )
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
            for prop in self.material.properties
            if prop.name in required
            and (
                prop.extra.get("scope") == "medium" or "scope" not in prop.extra
            )
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

    def validate_phase(self, phase: Phase) -> bool:
        return phase.validate()

    def get_phases(self) -> list[Phase]:
        """Returns all defined phases in order."""
        return [
            p
            for p in [self.solid, self.aqueous, self.nonaqueous, self.gas]
            if p
        ]

    def add_property(self, prop: MaterialProperty) -> None:
        self.properties.append(prop)

    def validate(self) -> bool:
        self._validate_required_phases()
        self._validate_extra_phases()
        self._validate_phase_objects()
        self._validate_required_properties()
        self._validate_extra_properties()
        return True

    def _validate_required_phases(self) -> None:
        required = [p["type"] for p in self.schema["phases"]]
        found = self._found_phase_types()
        missing = [ptype for ptype in required if ptype not in found]
        if missing:
            msg = f"Medium '{self.name}' is missing required phases: {missing}"
            raise ValueError(msg)

    def _validate_extra_phases(self) -> None:
        required = [p["type"] for p in self.schema["phases"]]
        found = self._found_phase_types()
        extra = [ptype for ptype in found if ptype not in required]
        if extra:
            msg = f"Medium '{self.name}' has unknown phases: {extra}"
            raise ValueError(msg)

    def _validate_phase_objects(self) -> None:
        for phase in self._found_phase_objects():
            phase.validate()

    def _validate_required_properties(self) -> None:
        required = self.schema.get("properties", [])
        found = [p.name for p in self.properties]
        missing = [p for p in required if p not in found]
        if missing:
            msg = f"Medium '{self.name}' is missing required properties: {missing}"
            raise ValueError(msg)

    def _validate_extra_properties(self) -> None:
        required = self.schema.get("properties", [])
        found = [p.name for p in self.properties]
        extra = [p for p in found if p not in required]
        if extra:
            msg = f"Medium '{self.name}' has unknown/unsupported properties: {extra}"
            raise ValueError(msg)

    def _found_phase_types(self) -> list[str]:
        return [
            phase.type
            for phase in [self.solid, self.aqueous, self.gas, self.nonaqueous]
            if phase is not None
        ]

    def _found_phase_objects(self) -> list:
        return [
            phase
            for phase in [self.solid, self.aqueous, self.gas, self.nonaqueous]
            if phase is not None
        ]

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        lines = [f"<Medium '{self.name}' (ID={self.material_id})>"]

        # Medium-level properties
        if self.properties:
            lines.append(f"  ├─ {len(self.properties)} medium properties:")
            for prop in self.properties:
                lines.append("  │   " + repr(prop))
        else:
            lines.append("  ├─ no medium-level properties")

        # Phases
        phases = self.get_phases()
        if phases:
            lines.append(f"  ├─ {len(phases)} phase(s):")
            for phase in phases:
                for line in repr(phase).splitlines():
                    lines.append("  │   " + line)
        else:
            lines.append("  └─ no phases defined")

        return "\n".join(lines)
