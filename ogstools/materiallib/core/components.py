from .component import Component
from .material import Material
from .property import Property


class Components:
    def __init__(
        self,
        phase_type: str,
        gas_component: Material,
        liquid_component: Material,
        process: str,
    ):
        self.phase_type = phase_type
        self.gas_properties: list[Property] = gas_component.get_properties()
        self.liquid_properties: list[Property] = (
            liquid_component.get_properties()
        )
        self.process = process

        self.gas_component = gas_component
        self.liquid_component = liquid_component

        self.gas_component_obj = self._create_component(self.gas_component, "A")
        self.liquid_component_obj = self._create_component(
            self.liquid_component, "W"
        )

    def _create_component(self, material: Material, role: str) -> Component:
        return Component(material, self.phase_type, role, self.process)

    def __repr__(self) -> str:
        return (
            f"<Components for phase '{self.phase_type}' with "
            f"Gas Component: {self.gas_component_obj.name}, "
            f"Liquid Component: {self.liquid_component_obj.name}.\n>"
            f"{self.gas_component_obj.name}: {self.gas_component_obj},\n>"
            f"{self.liquid_component_obj.name}: {self.liquid_component_obj},\n>"
        )
