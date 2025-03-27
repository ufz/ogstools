from pathlib import Path

import yaml

from ogstools.definitions import MATERIALS_DIR

from .material import RawMaterial


class MaterialDB:
    """
    Loads all material YAML files from the specified directory
    and converts them into `RawMaterial` objects.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path(MATERIALS_DIR)
        print(f"Loading materials from: {self.data_dir}")
        self.materials_db: dict[str, RawMaterial] = {}
        self._load_materials()

    def _load_materials(self) -> None:
        yaml_files = list(self.data_dir.glob("*.yml")) + list(
            self.data_dir.glob("*.yaml")
        )
        if not yaml_files:
            msg = f"No YAML files found in {self.data_dir}"
            raise FileNotFoundError(msg)

        for file_path in yaml_files:
            with file_path.open(encoding="utf-8") as file:
                raw_data = yaml.safe_load(file)
                name = raw_data.get("name", file_path.stem)
                material = RawMaterial(name=name, raw_data=raw_data)
                self.materials_db[name] = material

    def get_material(self, name: str) -> RawMaterial | None:
        """Returns a `RawMaterial` object by name."""
        return self.materials_db.get(name)

    def list_materials(self) -> list[str]:
        """Returns a list of all material names."""
        return list(self.materials_db.keys())

    def __repr__(self) -> str:
        return f"<MaterialDB with {len(self.materials_db)} materials from '{self.data_dir}'>"
