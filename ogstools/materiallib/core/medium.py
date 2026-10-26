# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import logging
from pathlib import Path
from typing import Any

from ogstools.definitions import MATERIALS_DIR
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

    :param material_id: The material ID assigned to this medium.
    :param material: The solid/medium material to use as property source.
    :param name: The name of the medium.
    :param fluids: A mapping from phase type to fluid material
        (e.g. `{"AqueousLiquid": water, "Gas": co2}`).
    :param process: The process type, e.g. "TH2M_PT". Used to determine
        the required structure.
    :param data_dir: The directory with the material data.
        `None` means `MATERIALS_DIR`.
    """

    def __init__(
        self,
        material_id: int,
        material: Material,
        name: str = "NO_NAME_GIVEN",
        fluids: dict[str, Material] | None = None,
        process: str = "",
        data_dir: Path | str | None = None,
    ):
        self.material_id = material_id
        self.material = material
        self.name = name
        self.data_dir = (
            Path(data_dir) if data_dir is not None else MATERIALS_DIR
        )

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
                        data_dir=self.data_dir,
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
                        if (
                            self.fluids.get("Gas") is None
                            or self.fluids.get("AqueousLiquid") is None
                        ):
                            msg = f"Both Gas and AqueousLiquid fluids required for {phase_type} with components."
                            raise ValueError(msg)

                        phase = Phase(
                            phase_type=phase_type,
                            gas_material=self.fluids.get("Gas"),
                            liquid_material=self.fluids.get("AqueousLiquid"),
                            process=self.process,
                            data_dir=self.data_dir,
                        )

                    # --- Case 1 + 2: No components ---
                    elif phase_type == "Gas":
                        if self.fluids.get("Gas") is None:
                            msg = 'Gas phase requires a "Gas" entry in fluids.'
                            raise ValueError(msg)
                        phase = Phase(
                            phase_type="Gas",
                            gas_material=self.fluids["Gas"],
                            process=self.process,
                            data_dir=self.data_dir,
                        )

                    elif phase_type == "AqueousLiquid":
                        if self.fluids.get("AqueousLiquid") is None:
                            msg = 'AqueousLiquid phase requires an "AqueousLiquid" entry in fluids.'
                            raise ValueError(msg)
                        phase = Phase(
                            phase_type="AqueousLiquid",
                            liquid_material=self.fluids["AqueousLiquid"],
                            process=self.process,
                            data_dir=self.data_dir,
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
            if prop.name in required and prop.extra.get("domain") == "medium"
        ]

        loaded = {prop.name for prop in self.properties}
        missing = required - loaded

        if missing:
            msg = f"Missing required medium properties in material '{self.material.name}': {missing}"
            raise ValueError(msg)

        logger.debug("Loaded %s medium-level properties", len(self.properties))
        logger.debug(self.properties)

    def set_phase(self, phase: Phase) -> None:
        """Assign a phase to the slot of its type.

        :param phase: The phase to assign. It replaces a phase of the same type.
        """
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
        """Return the phase of the given type.

        :param phase_type: "Solid", "Gas", "AqueousLiquid" or "NonAqueousLiquid".
        :returns: The phase, or `None` if the medium has no phase of this type.
        """
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
        """Check a phase against the process schema.

        Raises `ValueError` if the phase does not match the schema.

        :param phase: The phase to check.
        :returns: `True` if the phase is valid.
        """
        return phase.validate()

    def get_phases(self) -> list[Phase]:
        """Return the defined phases.

        :returns: The phases in the order solid, aqueous, non-aqueous, gas.
        """
        return [
            p
            for p in [self.solid, self.aqueous, self.nonaqueous, self.gas]
            if p
        ]

    def add_property(self, prop: MaterialProperty) -> None:
        """Add a medium-level property to this medium.

        :param prop: The property to add.
        """
        self.properties.append(prop)

    def validate(self) -> bool:
        """Check the phases and properties against the process schema.

        Raises `ValueError` for a missing or unknown phase or property.

        :returns: `True` if the medium is valid.
        """
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
