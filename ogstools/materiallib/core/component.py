# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging
from typing import Any

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .material import Material
from .property import MaterialProperty

logger = logging.getLogger(__name__)


class Component:
    def __init__(
        self,
        material: Material,
        phase_type: str,
        role: str,  # This materials role in the phase, e.g. 'solute' or 'solvent, etc.
        process: str,
        diffusion_coefficient: float,
    ):
        self.material = material
        self.phase_type = phase_type
        self.role = role
        self.name = material.name

        schema = PROCESS_SCHEMAS.get(process)
        if not schema:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)
        self.schema: dict[str, Any] = schema

        if self.phase_type == "Gas" and self.role == "Vapour":
            self.D = diffusion_coefficient
            logger.info(
                "Binary diffusion coefficient (Component '%s'): %s",
                self.name,
                self.D,
            )
        else:
            self.D = 0.0

        self.properties: list[MaterialProperty] = (
            self._get_filtered_properties()
        )

    def _get_filtered_properties(self) -> list[MaterialProperty]:
        """
        This method filters the material properties based on the process schema
        and the role (gas or liquid).
        """
        required_properties = set()

        logger.debug("===============================================")

        # Process schema check and filter required properties
        if self.schema:
            logger.debug(
                "Processing schema for phase type: [bold green]%s[/bold green], for [bold green]%s[/bold green] as: [bold green]%s[/bold green]",
                self.phase_type,
                self.name,
                self.role,
            )

            for phase_def in self.schema.get("phases", []):
                if phase_def.get("type") == self.phase_type:
                    logger.debug(
                        "Found phase definition for %s", self.phase_type
                    )

                    components = phase_def.get("components", {})
                    if self.role in components:
                        logger.debug(
                            "Found component role '%s': %s",
                            self.role,
                            components[self.role],
                        )
                        required_properties.update(components[self.role])

            filtered_properties = []

            for name in required_properties:
                if name == "diffusion":
                    logger.debug(
                        "Inserting binary diffusion coefficient for '%s': D = %s",
                        self.name,
                        self.D,
                    )
                    prop = MaterialProperty(
                        name="diffusion", type_="Constant", value=self.D
                    )
                    filtered_properties.append(prop)
                else:
                    for prop in self.material.properties:
                        if prop.name == name:
                            filtered_properties.append(prop)
                            break

        loaded = {prop.name for prop in filtered_properties}
        missing = required_properties - loaded

        if missing:
            msg = f"Missing required Component properties in material '{self.material.name}': {missing}"
            raise ValueError(msg)

        logger.debug("Loaded %s properties", len(filtered_properties))
        logger.debug(filtered_properties)

        logger.debug("===============================================\n")

        return filtered_properties

    def validate(self) -> bool:
        # Look up phase schema
        schema_phases = self.schema.get("phases", [])
        matching_phase = next(
            (p for p in schema_phases if p.get("type") == self.phase_type),
            None,
        )
        if not matching_phase:
            msg = f"Component '{self.name}' is in a phase '{self.phase_type}' not allowed for process."
            raise ValueError(msg)

        # Role check
        components_schema = matching_phase.get("components", {})
        if self.role not in components_schema:
            msg = f"Component '{self.name}' with role '{self.role}' not allowed in phase '{self.phase_type}'."
            raise ValueError(msg)

        # Allowed property names for this role
        allowed_props = set(
            components_schema[self.role] or []
        )  # ← may be empty
        actual_props = {p.name for p in self.properties}

        missing = allowed_props - actual_props
        extra = actual_props - allowed_props

        if missing or extra:
            msg = f"Component '{self.name}' in phase '{self.phase_type}' (role '{self.role}') is invalid.\n"
            if missing:
                msg += f"  Missing properties: {sorted(missing)}\n"
            if extra:
                msg += f"  Unexpected properties: {sorted(extra)}"
            raise ValueError(msg)

        return True

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        lines = [
            f"<Component '{self.name}' (Role: {self.role}, Phase: {self.phase_type})>"
        ]
        if self.properties:
            lines.append(f"  ├─ {len(self.properties)} properties:")
            for prop in self.properties:
                lines.append("  │   " + repr(prop))
        else:
            lines.append("  └─ no properties")
        return "\n".join(lines)
