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
        self.schema: dict[str, Any] | None = PROCESS_SCHEMAS.get(process)

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

        self.properties: list[Property] = []
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
        print(f"Loading properties for phase type: {self.type}")
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

        print(f"Found phase definition for {self.type}")
        required = set(phase_def.get("properties", []))
        print(f"Required properties: {required}")

        source = {
            "AqueousLiquid": self.liquid_material,
            "Gas": self.gas_material,
            "Solid": self.solid_material,
        }.get(self.type)

        if source is None:
            msg = f"Don't know how to load properties for phase type '{self.type}'"
            raise ValueError(msg)

        print(f"Source material: {source.name}")

        self.properties = [
            prop for prop in source.get_properties() if prop.name in required
        ]

        loaded = {prop.name for prop in self.properties}
        missing = required - loaded

        if missing:
            msg = f"Missing required properties for phase type '{self.type}', material '{source.name}': {missing}"
            raise ValueError(msg)

        print(
            f"Loaded {len(self.properties)} properties for phase type '{self.type}'"
        )
        print(self.properties)

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
