# ogstools/materiallib/__init__.py
from .io.yaml_loader import load_all_yaml

class MaterialLib:
    def __init__(self):
        self.materials = {}
        self._load_materials()

    def _load_materials(self):
        yaml_data = load_all_yaml()
        for filename, data in yaml_data.items():
            material_name = data.get("name", filename)
            self.materials[material_name] = data

    def list_materials(self):
        return list(self.materials.keys())

    def get_material(self, name: str):
        return self.materials.get(name)
