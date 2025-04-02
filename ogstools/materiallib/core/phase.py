from typing import Any

from rich import print

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .component import Component
from .components import Components
from .material import Material
from .property import Property


class Phase:
    def __init__(
        self,
        phase_type: str,
        gas_material: Material,
        liquid_material: Material,
        process: str,
    ):
        self.type = phase_type
        self.process = process
        self.gas_material = gas_material
        self.liquid_material = liquid_material
        self.schema: dict[str, Any] | None = PROCESS_SCHEMAS.get(process)

        self.properties: list[Property] = []
        self.components: list[Component] = []

        if not self.schema:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)

        self._load_phase_properties()

        if any(
            "components" in p
            for p in self.schema.get("phases", [])
            if p.get("type") == self.type
        ):
            self._load_components(gas_material, liquid_material)

    def _load_phase_properties(self) -> None:
        print(f"Loading properties for phase type: {self.type}")
        assert self.schema is not None
        for phase_def in self.schema.get("phases", []):
            if phase_def.get("type") == self.type:
                print(f"Found phase definition for {self.type}")
                required = set(phase_def.get("properties", []))
                print(f"Required properties: {required}")
                if self.type == "AqueousLiquid":
                    source = self.liquid_material
                elif self.type == "Gas":
                    source = self.gas_material
                else:
                    msg = f"Don't know how to load properties for phase type '{self.type}'"
                    raise ValueError(msg)

                self.properties = [
                    prop
                    for prop in source.get_properties()
                    if prop.name in required
                ]
                print(
                    f"Loaded {len(self.properties)} properties for phase type '{self.type}'"
                )
                print(self.properties)
                break

    def _load_components(
        self, gas_material: Material, liquid_material: Material
    ) -> None:
        comps = Components(
            phase_type=self.type,
            gas_component=gas_material,
            liquid_component=liquid_material,
            process=self.process,
        )
        self.components = [comps.gas_component_obj, comps.liquid_component_obj]

    def add_property(self, prop: Property) -> None:
        self.properties.append(prop)

    def add_component(self, component: Component) -> None:
        self.components.append(component)

    def __repr__(self) -> str:
        return (
            f"<Phase '{self.type}' with {len(self.properties)} properties and "
            f"{len(self.components)} components>"
        )
