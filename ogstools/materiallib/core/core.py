from pathlib import Path
import yaml

from ogstools.definitions import MATERIALS_DIR

class Property:
    def __init__(self, name: str, type_: str, value=None, **extra):
    #def __init__(self, name: str, type_: str, value: float | None = None, **extra):
        self.name = name
        self.type = type_
        self.value = value
        self.extra = extra  # e.g. unit, slope, source, ...

    def to_dict(self):
        d = {"name": self.name, "type": self.type}
        if self.value is not None:
            d["value"] = self.value
        d.update(self.extra)
        return d


class Medium:
    def __init__(self, id_, name, properties, process="TH2M"):
        self.id = id_
        self.name = name
        self.properties = properties
        self.process = process

    def to_ogs_properties(self) -> list[dict]:
        return [prop.to_ogs_dict() for prop in self.properties]

    def add_to_project(self, prj):
        from ogstools.materiallib.integrator import MaterialIntegrator
        MaterialIntegrator(prj, process=self.process).add_medium(self)

class MaterialLib:
    def __init__(self, data_dir: Path = None, process: str = "TH2M"):
        self.materials = {}
        self.data_dir = data_dir or Path(__file__).parents[1] / "data"
        self.process = process  # save OGS-process
class Material:
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
            if not isinstance(entries, list):
                entries = [entries]
            for entry in entries:
                type_ = entry.get("type", "Constant")
                value = entry.get("value", None)
                extra = {k: v for k, v in entry.items() if k not in ("type", "value")}
                prop = Property(name=prop_name, type_=type_, value=value, **extra)
                self.properties.append(prop)

    def get_property_names(self) -> list[str]:
        return [p.name for p in self.properties]

    def get_properties(self) -> list:
        return self.properties

    def __repr__(self):
        return f"<Material '{self.name}' with {len(self.properties)} properties>"

    from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS
from ogstools.materiallib.validation import validate_medium

class MaterialDB:
    """
    Loads all material YAML files from the specified directory
    and converts them into `Material` objects.
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(MATERIALS_DIR)
        print(f"Loading materials from: {self.data_dir}")
        self.materials_db = {}
        self._load_materials()

    def _load_materials(self):
        yaml_files = list(self.data_dir.glob("*.yml")) + list(self.data_dir.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML files found in {self.data_dir}")

        for file_path in yaml_files:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_data = yaml.safe_load(file)
                name = raw_data.get("name", file_path.stem)
                material = Material(name=name, raw_data=raw_data)
                self.materials_db[name] = material

    def get_material(self, name: str) -> Material | None:
        """Returns a `Material` object by name."""
        return self.materials_db.get(name)

    def list_materials(self) -> list[str]:
        """Returns a list of all material names."""
        return list(self.materials_db.keys())

    def __repr__(self):
        return f"<MaterialDB with {len(self.materials_db)} materials from '{self.data_dir}'>"

    def create_medium(self, name: str, id_: int, overrides: dict = None) -> Medium:
        data = self.get_material(name)
        if data is None:
            raise ValueError(f"Material '{name}' not found in database.")

        # Apply flat or shallow overrides
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(data.get(key), dict):
                    data[key] = {**data[key], **value}  # safer copy
                else:
                    data[key] = value

        props = []
        for key, block in data.items():
            if key.endswith("_properties"):
                for pname, plist in block.items():
                    if not isinstance(plist, list):
                        plist = [plist]

                    entry = plist[0]  # Only use the first type for now
                    type_ = entry.get("type", "Constant")
                    value = entry.get("value")
                    extras = {k: v for k, v in entry.items() if k not in ["type", "value"]}
                    props.append(Property(name=pname, type_=type_, value=value, **extras))

        return Medium(id_=id_, name=name, properties=props, process=self.process)
