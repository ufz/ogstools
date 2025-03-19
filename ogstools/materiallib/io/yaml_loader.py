# io/yaml_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any

# Standard-Pfad für YAML-Daten
DATA_DIR = Path(__file__).parent.parent / "data"

def load_all_yaml() -> Dict[str, Any]:
    """
    Loads all YAML files from the data directory.

    :return: Dictionary with filenames as keys and YAML content as values
    :raises FileNotFoundError: If no YAML files are found
    """
    yaml_files = list(DATA_DIR.glob("*.yml")) + list(DATA_DIR.glob("*.yaml"))

    if not yaml_files:
        raise FileNotFoundError("No YAML files found in the data directory.")

    all_data = {}
    for file_path in yaml_files:
        with open(file_path, "r", encoding="utf-8") as file:
            all_data[file_path.name] = yaml.safe_load(file)

    return all_data

if __name__ == "__main__":
    # Test: Alle YAML-Dateien laden
    try:
        all_yaml_data = load_all_yaml()
        print("Loaded YAML files:")
        for filename, content in all_yaml_data.items():
            print(f" - {filename}: {content}")
    except FileNotFoundError as e:
        print(e)
