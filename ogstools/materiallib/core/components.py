# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path

import yaml  # type: ignore[import]

# from ogstools.definitions import MATERIALS_DIR
import ogstools.definitions as defs

from .component import Component
from .material import Material
from .property import MaterialProperty


class Components:

    def __init__(
        self,
        phase_type: str,
        gas_component: Material,
        liquid_component: Material,
        process: str,
        Diffusion_coefficient: float | None = None,
    ):
        self.phase_type = phase_type
        self.gas_properties: list[MaterialProperty] = gas_component.properties
        self.liquid_properties: list[MaterialProperty] = (
            liquid_component.properties
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
            Diffusion_coefficient
            if Diffusion_coefficient is not None
            else self.get_binary_diffusion_coefficient(self.phase_type)
        )

        self.gas_component_obj = self._create_component(
            self.gas_component, gas_role, D
        )
        self.liquid_component_obj = self._create_component(
            self.liquid_component, liquid_role, D
        )

    def get_binary_diffusion_coefficient(self, phase: str) -> float:
        """
        Retrieves the binary diffusion coefficient D [m²/s] for a specific component pair
        in either the liquid or gas phase.

        In the liquid phase, this typically describes a gas (e.g. CO2) diffusing in a solvent
        like water. In the gas phase, it typically describes a vapor (e.g. H2O) diffusing in a
        carrier gas like CO₂.

        Parameters:
            phase (str): Either "liquid" or "gas". Indicates the phase in which diffusion
                        occurs.

        Returns:
            float: Binary diffusion coefficient in [m²/s].

        Raises:
            ValueError: If the component pair is not found.
        """
        if phase == "AqueousLiquid":
            # Solute (gas component) diffuses in solvent (liquid component)
            solvent = self.liquid_component.name
            solute = self.gas_component.name
        elif phase == "Gas":
            # Solute (liquid component) evaporates into solvent (gas component)
            solvent = self.gas_component.name
            solute = self.liquid_component.name
        else:
            msg = f"Invalid phase '{phase}'. Must be 'AqueousLiquid' or 'Gas'."
            raise ValueError(msg)

        file_path = Path(defs.MATERIALS_DIR) / "diffusion_coefficients.yml"
        with file_path.open() as f:
            data = yaml.safe_load(f)

        try:
            return float(data[phase][solvent][solute])
        except KeyError:
            pass

        try:
            return float(data[phase][solute][solvent])
        except KeyError as err:
            msg = (
                f"No {phase}-phase diffusion coefficient found for the pair "
                f"'{solvent} / {solute}' in {file_path}"
            )
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

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        lines = [f"<Components for phase '{self.phase_type}'>"]
        for comp in [self.gas_component_obj, self.liquid_component_obj]:
            for line in repr(comp).splitlines():
                lines.append("  " + line)
        return "\n".join(lines)
