from .property import Property


class RawMaterial:
    """
    Represents a single material, parsed from YAML data.
    """

    def __init__(self, name: str, raw_data: dict):
        self.name = name
        self.raw = raw_data  # full YAML (e.g. for debugging or export)
        self.properties = []

        self._parse_properties()

    def _parse_properties(self):
        block = self.raw.get("properties", {})
        for prop_name, entries in block.items():
            for entry in entries if isinstance(entries, list) else [entries]:
                type_ = entry.get("type", "Constant")
                value = entry.get("value", None)
                extra = {
                    k: v for k, v in entry.items() if k not in ("type", "value")
                }
                prop = Property(
                    name=prop_name, type_=type_, value=value, **extra
                )
                self.properties.append(prop)

    def get_property_names(self) -> list[str]:
        return [p.name for p in self.properties]

    def get_properties(self) -> list:
        return self.properties

    def __repr__(self):
        return f"<RawMaterial '{self.name}' with {len(self.properties)} properties>"


class Material:
    """
    A validated and filtered material object ready to be used
    in OGS media and phases.
    """

    def __init__(self, name: str, properties: list[Property]):
        self.name = name
        self.properties = properties

    def get_properties(self):
        return self.properties

    def get_property_names(self):
        return [p.name for p in self.properties]

    def __repr__(self):
        return (
            f"<Material '{self.name}' with {len(self.properties)} properties>"
        )
