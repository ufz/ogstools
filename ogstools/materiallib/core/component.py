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

        self.schema = PROCESS_SCHEMAS.get(process)
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

        # Process schema check and filter required properties
        if self.schema:
            print(
                f"Processing schema for phase type: {self.phase_type}"
            )  # Debug: Phase being processed

            # Iterate over the process schema to find the relevant properties
            for key, value in self.schema.items():
                print(
                    f"Checking key: {key}, value: {value}"
                )  # Debug: Output the schema's current key and value

                # Ensure 'value' is a dictionary and the phase matches the current phase type
                if isinstance(value, dict) and key == self.phase_type:
                    # Debug output for components in the phase
                    if "Components" in value:
                        print(
                            f"Components found for {key}: {value['Components']}"
                        )  # Debug: Show components

                        # Now check the role within the components
                        if self.role in value["Components"]:
                            # Debug: Role found, print properties
                            print(
                                f"Properties for role '{self.role}': {value['Components'][self.role]}"
                            )

                            # Add the properties for the current role (gas or liquid)
                            required_properties.update(
                                value["Components"][self.role]
                            )
                        else:
                            print(
                                f"Role '{self.role}' not found in components for {key}"
                            )  # Debug: Role not found
                    else:
                        print(
                            f"No components found in schema for phase {key}"
                        )  # Debug: No components
                else:
                    print(
                        f"Skipping phase {key} because it does not match {self.phase_type} or is not a dict"
                    )  # Debug: Phase mismatch

        # Debug: Print the required properties
        print(f"Required properties: {required_properties}")

        # Now filter the material properties based on the schema
        filtered_properties = [
            prop
            for prop in self.material.get_properties()
            if prop.name in required_properties
        ]

        # Debug: Print filtered properties
        print(f"Filtered properties: {filtered_properties}")

        return filtered_properties

    # def _get_filtered_properties(self) -> list[Property]:
    #     """
    #     This method filters the material properties based on the process schema
    #     and the role (gas or liquid).
    #     """
    #     required_properties = set()

    #     # Process schema check and filter required properties
    #     if self.schema:
    #         # Here we filter the properties that are specified in the process schema
    #         # for this phase and role
    #         for key, value in self.schema.items():
    #             # Make sure `value` is a dictionary and the role is in the components
    #             if (
    #                 isinstance(value, dict)
    #                 and key == self.phase_type
    #                 and "Components" in value
    #                 and self.role in value["Components"]
    #             ):
    #                 # Add all properties for the role (gas or liquid)
    #                 required_properties.update(value["Components"][self.role])
    #                 print(value["Components"][self.role])
    #                 print("##############################")

    #     # Now we filter the material properties based on the schema
    #     return [
    #         prop
    #         for prop in self.material.get_properties()
    #         if prop.name in required_properties
    #     ]

    def __repr__(self) -> str:
        # Sammle die Namen der Eigenschaften
        property_names = [prop.name for prop in self.properties]

        # Gebe die Namen der Eigenschaften aus
        return (
            f"<Component '{self.name}' (Role: {self.role}, Phase: {self.phase_type}) with "
            f"{len(self.properties)} properties:\n"
            f"  ├─ {'\n  ├─ '.join(property_names)}>"
        )
