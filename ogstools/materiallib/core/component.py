from typing import Any

from rich import print

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .material import Material
from .property import Property


class Component:
    def __init__(
        self,
        material: Material,
        phase_type: str,
        role: str,  # This materials role in the phase, e.g. 'solute' or 'solvent, etc.
        process: str,
    ):
        self.material = material
        self.phase_type = phase_type
        self.role = role
        self.name = material.name

        self.schema: dict[str, Any] | None = PROCESS_SCHEMAS.get(process)
        if not self.schema:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)

        self.properties: list[Property] = self._get_filtered_properties()

    def _get_filtered_properties(self) -> list[Property]:
        """
        This method filters the material properties based on the process schema
        and the role (gas or liquid).
        """
        required_properties = set()

        print("===============================================\n")

        # Process schema check and filter required properties
        if self.schema:
            print(
                f"Processing schema for phase type: [bold green]{self.phase_type}[/bold green], for [bold green]{self.name}[/bold green] as: [bold green]{self.role}[/bold green]"
            )

            for phase_def in self.schema.get("phases", []):
                if phase_def.get("type") == self.phase_type:
                    print(f"Found phase definition for {self.phase_type}")

                    # Add general phase properties
                    required_properties.update(phase_def.get("properties", []))

                    # Optional: add component-specific properties
                    components = phase_def.get("components", {})
                    if self.role in components:
                        print(
                            f"Found component role '{self.role}': {components[self.role]}"
                        )
                        required_properties.update(components[self.role])

        filtered_properties = [
            prop
            for prop in self.material.get_properties()
            if prop.name in required_properties
        ]

        print("\n===============================================\n")

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
