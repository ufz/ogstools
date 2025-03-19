# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging
from typing import Any

from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

from .component import Component
from .components import Components
from .material import Material
from .property import MaterialProperty

logger = logging.getLogger(__name__)


class Phase:
    def __init__(
        self,
        phase_type: str,
        gas_material: Material | None = None,
        liquid_material: Material | None = None,
        solid_material: Material | None = None,
        process: str = "",
    ):
        self.type = phase_type
        self.process = process
        self.gas_material = gas_material
        self.liquid_material = liquid_material
        self.solid_material = solid_material

        schema = PROCESS_SCHEMAS.get(process)
        if not schema:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)
        self.schema: dict[str, Any] = schema

        # Check for material consistency
        match phase_type:
            case "AqueousLiquid":
                if liquid_material is None:
                    raise ValueError(
                        ("AqueousLiquid requires liquid_material.",)[0]
                    )
                if solid_material is not None:
                    raise ValueError(
                        ("AqueousLiquid must not have solid_material.",)[0]
                    )
            case "Gas":
                if gas_material is None:
                    raise ValueError(("Gas requires gas_material.",)[0])
                if solid_material is not None:
                    raise ValueError(("Gas must not have solid_material.",)[0])
            case "Solid":
                if solid_material is None:
                    raise ValueError(("Solid requires solid_material.",)[0])
                if gas_material is not None or liquid_material is not None:
                    raise ValueError(
                        (
                            "Solid must not have gas_material or liquid_material.",
                        )[0]
                    )

        self.properties: list[MaterialProperty] = []
        self.components: list[Component] = []

        if not self.schema:
            msg = f"No process schema found for '{process}'."
            raise ValueError(msg)

        self._load_phase_properties()

        if (
            gas_material is not None
            and liquid_material is not None
            and any(
                "components" in p
                for p in self.schema.get("phases", [])
                if p.get("type") == self.type
            )
        ):
            self._load_components(gas_material, liquid_material)

    def _load_phase_properties(self) -> None:
        logger.debug("Loading properties for phase type: %s", self.type)
        assert self.schema is not None

        phase_def = next(
            (
                p
                for p in self.schema.get("phases", [])
                if p.get("type") == self.type
            ),
            None,
        )

        if phase_def is None:
            msg = f"No phase definition found for type '{self.type}'"
            raise ValueError(msg)

        logger.debug("Found phase definition for %s", self.type)
        required = set(phase_def.get("properties", []))
        logger.debug("Required properties: %s", required)

        source = {
            "AqueousLiquid": self.liquid_material,
            "Gas": self.gas_material,
            "Solid": self.solid_material,
        }.get(self.type)

        if source is None:
            msg = f"Don't know how to load properties for phase type '{self.type}'"
            raise ValueError(msg)

        logger.debug("Source material: %s", source.name)

        self.properties = [
            prop
            for prop in source.properties
            if prop.name in required
            and (
                prop.extra.get("scope") == "phase" or "scope" not in prop.extra
            )
        ]

        loaded = {prop.name for prop in self.properties}
        missing = required - loaded

        if missing:
            msg = f"Missing required properties for phase type '{self.type}', material '{source.name}': {missing}"
            raise ValueError(msg)

        logger.debug(
            "Loaded %s properties for phase type '%s'",
            len(self.properties),
            self.type,
        )
        logger.debug(self.properties)

    def _load_components(
        self,
        gas_material: Material,
        liquid_material: Material,
        Diffusion_coefficient: float | None = None,
    ) -> None:
        self.components_obj = Components(
            phase_type=self.type,
            gas_component=gas_material,
            liquid_component=liquid_material,
            process=self.process,
            Diffusion_coefficient=Diffusion_coefficient,
        )
        self.components = [
            self.components_obj.gas_component_obj,
            self.components_obj.liquid_component_obj,
        ]

    def add_property(self, prop: MaterialProperty) -> None:
        self.properties.append(prop)

    def add_component(self, component: Component) -> None:
        self.components.append(component)

    def validate(self) -> bool:
        self._validate_phase_exists()
        self._validate_required_properties()
        self._validate_extra_properties()
        self._validate_components()
        return True

    def _validate_phase_exists(self) -> None:
        if not any(p["type"] == self.type for p in self.schema["phases"]):
            msg = f"Phase '{self.type}' is not defined for this process."
            raise ValueError(msg)

    def _validate_required_properties(self) -> None:
        required = self._required_properties()
        found = self._found_property_names()
        missing = [p for p in required if p not in found]
        if missing:
            msg = (
                f"Phase '{self.type}' is missing required properties: {missing}"
            )
            raise ValueError(msg)

    def _validate_extra_properties(self) -> None:
        required = self._required_properties()
        found = self._found_property_names()
        extra = [p for p in found if p not in required]
        if extra:
            msg = f"Phase '{self.type}' has unknown/unsupported properties: {extra}"
            raise ValueError(msg)

    def _validate_components(self) -> None:
        for component in self.components:
            if not component.validate():
                msg = f"Component '{component.name}' in phase '{self.type}' is invalid."
                raise ValueError(msg)

    def _required_properties(self) -> list[str]:
        for p in self.schema["phases"]:
            if p["type"] == self.type:
                return p.get("properties", [])
        return []

    def _found_property_names(self) -> list[str]:
        return [p.name for p in self.properties]

    # -----------------------
    # Representation
    # -----------------------
    def __repr__(self) -> str:
        lines = [f"<Phase '{self.type}'>"]

        if self.properties:
            lines.append(f"  ├─ {len(self.properties)} properties:")
            for prop in self.properties:
                lines.append("  │   " + repr(prop))
        else:
            lines.append("  ├─ no properties")

        if self.components:
            lines.append(f"  ├─ {len(self.components)} component groups:")
            for comp in self.components:
                for line in repr(comp).splitlines():
                    lines.append("  │   " + line)
        else:
            lines.append("  └─ no components")

        return "\n".join(lines)
