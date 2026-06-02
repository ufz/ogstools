# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import]

from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.material_manager import MaterialManager


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(filename: str, data: dict) -> Path:
        path = tmp_path / filename
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        return path

    return _write


def test_material_from_file_accepts_grouped_domains(write_yaml) -> None:
    file_path = write_yaml(
        "granite.yml",
        {
            "name": "granite",
            "domains": [
                {
                    "domain": "medium",
                    "properties": {
                        "Density": [{"type": "Constant", "value": 2700}]
                    },
                }
            ],
        },
    )

    material = Material.from_file(file_path)

    assert material is not None
    assert material.name == "granite"
    assert "Density" in material
    assert material["Density"].extra["domain"] == "medium"


def test_material_to_file_roundtrip_preserves_grouped_domains(
    tmp_path: Path, write_yaml
) -> None:
    file_path = write_yaml(
        "water.yml",
        {
            "name": "water",
            "domains": [
                {
                    "domain": "phase",
                    "properties": {
                        "Viscosity": [{"type": "Constant", "value": 1.0}]
                    },
                }
            ],
        },
    )

    material = Material.from_file(file_path)
    assert material is not None

    target = tmp_path / "water_copy.yml"
    material.to_file(target)

    copied_raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "domains" in copied_raw
    assert "properties" not in copied_raw
    assert copied_raw["domains"][0]["domain"] == "phase"


def test_material_rejects_flat_top_level_properties(write_yaml) -> None:
    file_path = write_yaml(
        "legacy.yml",
        {
            "name": "legacy",
            "properties": {"Density": {"type": "Constant", "value": 2700}},
        },
    )

    with pytest.raises(ValueError, match="top-level 'domains'"):
        Material.from_file(file_path)


def test_material_rejects_duplicate_domain_blocks(write_yaml) -> None:
    file_path = write_yaml(
        "duplicate.yml",
        {
            "name": "duplicate",
            "domains": [
                {"domain": "medium", "properties": {}},
                {"domain": "medium", "properties": {}},
            ],
        },
    )

    with pytest.raises(ValueError, match="duplicate top-level domain block"):
        Material.from_file(file_path)


def test_material_rejects_invalid_domain_name(write_yaml) -> None:
    file_path = write_yaml(
        "invalid.yml",
        {
            "name": "invalid",
            "domains": [
                {"domain": "fluid", "properties": {}},
            ],
        },
    )

    with pytest.raises(ValueError, match="unsupported domain"):
        Material.from_file(file_path)


def test_material_manager_loads_grouped_domain_repository(
    tmp_path: Path, write_yaml
) -> None:
    write_yaml(
        "granite.yml",
        {
            "name": "granite",
            "domains": [
                {
                    "domain": "medium",
                    "properties": {
                        "Density": [{"type": "Constant", "value": 2700}]
                    },
                }
            ],
        },
    )

    manager = MaterialManager(data_dir=tmp_path)

    assert "granite" in manager.materials_db
