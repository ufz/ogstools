# ogstools/materiallib/core.py

from pathlib import Path
import yaml


class Property:
    def __init__(self, name: str, type_: str, value: float | None = None, **extra):
        self.name = name
        self.type = type_
        self.value = value
        self.extra = extra  # weitere Schlüssel wie slope, etc.

    def to_ogs_dict(self) -> dict:
        d = {"name": self.name, "type": self.type}
        if self.value is not None:
            d["value"] = self.value
        d.update(self.extra)
        return d


class Medium:
    def __init__(self, id_: int, name: str, properties: list[Property]):
        self.id = id_
        self.name = name
        self.properties = properties

    def to_ogs_properties(self) -> list[dict]:
        return [prop.to_ogs_dict() for prop in self.properties]

    def add_to_project(self, prj):
        prj.media.add_medium(
            material_id=self.id,
            properties=self.to_ogs_properties()
        )


class MaterialLib:
    def __init__(self, data_dir: Path = None):
        self.materials = {}
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self._load_materials()

    def _load_materials(self):
        yaml_files = list(self.data_dir.glob("*.yml")) + list(self.data_dir.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML files found in the data directory ({self.data_dir}).")
        for file_path in yaml_files:
            with open(file_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
                name = data.get("name", file_path.stem)
                self.materials[name] = data

    def get_material(self, name: str) -> dict | None:
        return self.materials.get(name)

    def create_medium(self, name: str, id_: int) -> Medium:
        data = self.get_material(name)
        if data is None:
            raise ValueError(f"Material '{name}' not found in database.")

        props_raw = data.get("properties", {})
        props = []

        for pname, plist in props_raw.items():
            if not isinstance(plist, list):
                plist = [plist]  # YAML-Einträge, die kein list sind
            for entry in plist:
                type_ = entry.get("type")
                value = entry.get("value")
                extras = {k: v for k, v in entry.items() if k not in ["type", "value"]}
                props.append(Property(name=pname, type_=type_, value=value, **extras))

        return Medium(id_=id_, name=name, properties=props)
