# ogstools/materiallib/__init__.py
import yaml
from pathlib import Path

class MaterialLib:
    def __init__(self):
        self.materials = {}
        self._load_materials()

    def _load_materials(self):
        data_dir = Path(__file__).parent.parent / "data"
        yaml_files = list(data_dir.glob("*.yml")) + list(data_dir.glob("*.yaml"))

        if not yaml_files:
            raise FileNotFoundError(f"No YAML files found in the data directory ({data_dir}).")

        self.materials = {}
        for file_path in yaml_files:
            with open(file_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
                material_name = data.get("name", file_path.stem)
                print(f"MAterial found: {material_name}")
                self.materials[material_name] = data

    def list_materials(self):
        return list(self.materials.keys())

    def get_material(self, name: str):
        return self.materials.get(name)
