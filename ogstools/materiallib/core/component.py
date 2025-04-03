import logging
from typing import Any

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .material import Material
from .property import Property

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

        self.schema: dict[str, Any] | None = PROCESS_SCHEMAS.get(process)
        if not self.schema:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)

        if self.phase_type == "Gas" and self.role == "Vapour":
            self.D = diffusion_coefficient
            logger.info(
                "Binary diffusion coefficient (Component '%s'): %s",
                self.name,
                self.D,
            )
        else:
            self.D = 0.0

        self.properties: list[Property] = self._get_filtered_properties()

    def _get_filtered_properties(self) -> list[Property]:
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
                    prop = Property(
                        name="diffusion", type_="Constant", value=self.D
                    )
                    filtered_properties.append(prop)
                else:
                    for prop in self.material.get_properties():
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

    def __repr__(self) -> str:
        # Sammle die Namen der Eigenschaften
        property_names = [prop.name for prop in self.properties]

        # Gebe die Namen der Eigenschaften aus
        return (
            f"<Component '{self.name}' (Role: {self.role}, Phase: {self.phase_type}) with "
            f"{len(self.properties)} properties:\n"
            f"  ├─ {'\n  ├─ '.join(property_names)}>"
        )
