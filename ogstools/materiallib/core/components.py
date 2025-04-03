from pathlib import Path

import yaml

from ogstools.definitions import MATERIALS_DIR

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

        if self.phase_type == "AqueousLiquid":
            gas_role = "Solute"
            liquid_role = "Solvent"
        elif self.phase_type == "Gas":
            gas_role = "Carrier"
            liquid_role = "Vapour"
        else:
            msg = f"Unsupported phase_type: {self.phase_type}"
            raise ValueError(msg)

        D = (
            self.get_binary_diffusion_coefficient()
            if self.phase_type == "Gas"
            else 0.0
        )

        self.gas_component_obj = self._create_component(
            self.gas_component, gas_role, D
        )
        self.liquid_component_obj = self._create_component(
            self.liquid_component, liquid_role, D
        )

    def get_binary_diffusion_coefficient(self) -> float:

        a = self.gas_component.name
        b = self.liquid_component.name

        file_path = Path(MATERIALS_DIR) / "diffusion_coefficients.yml"

        with Path.open(file_path) as f:
            data = yaml.safe_load(f)

        # first try A→B
        try:
            return float(data[a][b])
        except KeyError:
            pass

        # then try B→A
        try:
            return float(data[b][a])
        except KeyError as err:
            msg = f"No diffusion coefficient found for pair '{a} / {b}' in {file_path}"
            raise ValueError(msg) from err

    def _create_component(
        self, material: Material, role: str, D: float
    ) -> Component:
        return Component(
            material,
            self.phase_type,
            role,
            self.process,
            diffusion_coefficient=D,
        )

    def __repr__(self) -> str:
        return (
            f"<Components for phase '{self.phase_type}' with "
            f"Gas Component: {self.gas_component_obj.name}, "
            f"Liquid Component: {self.liquid_component_obj.name}.\n>"
            f"{self.gas_component_obj.name}: {self.gas_component_obj},\n>"
            f"{self.liquid_component_obj.name}: {self.liquid_component_obj},\n>"
        )
