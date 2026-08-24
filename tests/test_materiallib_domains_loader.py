# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import]

from ogstools.definitions import ROOT_DIR
from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.material_manager import MaterialManager
from ogstools.materiallib.core.property import ParameterValue
from ogstools.materiallib.distributions import UniformDistribution

EXAMPLES_DIR = ROOT_DIR / "examples" / "materiallib"


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(filename: str, data: dict) -> Path:
        path = tmp_path / filename
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        return path

    return _write


@pytest.fixture
def copy_example_material(tmp_path: Path):
    def _copy(filename: str) -> Path:
        source = EXAMPLES_DIR / filename
        target = tmp_path / filename
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    return _copy


def test_material_from_file_accepts_grouped_domains() -> None:
    file_path = EXAMPLES_DIR / "opalinus.yml"

    material = Material.from_file(file_path)

    assert material is not None
    assert material.name == "opalinus_clay"
    assert "porosity" in material
    assert material["porosity"].extra["domain"] == "medium"


def test_material_to_file_roundtrip_preserves_grouped_domains(
    tmp_path: Path,
) -> None:
    file_path = EXAMPLES_DIR / "water.yml"

    material = Material.from_file(file_path)
    assert material is not None

    target = tmp_path / "water_copy.yml"
    material.to_file(target)

    copied_raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "domains" in copied_raw
    assert "properties" not in copied_raw
    assert copied_raw["domains"][0]["domain"] == "phase"


def test_material_parses_wrapped_parameter_value() -> None:
    file_path = EXAMPLES_DIR / "distributed_demo.yml"

    material = Material.from_file(file_path)

    value = material["porosity"].parameters["value"]
    assert value == ParameterValue(
        base_value=0.15,
        distribution=UniformDistribution(lower=0.10, upper=0.20),
    )


def test_material_preserves_wrapped_parameter_value_after_raw_rebuild(
    tmp_path: Path,
) -> None:
    file_path = EXAMPLES_DIR / "distributed_demo.yml"

    material = Material.from_file(file_path)
    material.filter_properties("storage")

    target = tmp_path / "distributed_demo_copy.yml"
    material.to_file(target)

    copied_raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert copied_raw["domains"][0]["properties"]["storage"]["value"] == {
        "base_value": 2.0e-10
    }


def test_material_rejects_unsupported_top_level_properties_key(
    write_yaml,
) -> None:
    file_path = write_yaml(
        "legacy.yml",
        {
            "name": "legacy",
            "properties": {"Density": {"type": "Constant", "value": 2700}},
        },
    )

    with pytest.raises(ValueError, match=r"top-level key.*properties"):
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
    tmp_path: Path, copy_example_material
) -> None:
    copy_example_material("opalinus.yml")

    manager = MaterialManager(data_dir=tmp_path)

    assert "opalinus_clay" in manager.materials_db


def test_material_medium_property_accessor_returns_medium_property() -> None:
    file_path = EXAMPLES_DIR / "distributed_demo.yml"

    material = Material.from_file(file_path)

    assert material.medium.property("density").parameters["value"] == 2700


def test_material_phase_property_accessor_returns_phase_property() -> None:
    file_path = EXAMPLES_DIR / "distributed_demo.yml"

    material = Material.from_file(file_path)

    assert material.phase.property("density").parameters["value"] == 999


def test_material_component_property_accessor_returns_component_property() -> (
    None
):
    file_path = EXAMPLES_DIR / "water.yml"

    material = Material.from_file(file_path)

    assert (
        material.component.property("molar_mass").parameters["value"]
        == 0.018016
    )


def test_material_property_parameter_returns_plain_scalar_value() -> None:
    file_path = EXAMPLES_DIR / "opalinus.yml"

    material = Material.from_file(file_path)

    assert material.medium.property("porosity").parameter("value") == 0.15


def test_material_property_parameter_returns_wrapped_parameter_value() -> None:
    file_path = EXAMPLES_DIR / "distributed_demo.yml"

    material = Material.from_file(file_path)

    assert material.medium.property("porosity").parameter(
        "value"
    ) == ParameterValue(
        base_value=0.15,
        distribution=UniformDistribution(lower=0.10, upper=0.20),
    )


def test_material_property_parameter_rejects_missing_parameter() -> None:
    file_path = EXAMPLES_DIR / "opalinus.yml"

    material = Material.from_file(file_path)

    with pytest.raises(
        KeyError, match="Property porosity has no parameter called 'missing'"
    ):
        material.medium.property("porosity").parameter("missing")


def test_material_property_accessor_rejects_missing_property_in_domain() -> (
    None
):
    file_path = EXAMPLES_DIR / "water.yml"

    material = Material.from_file(file_path)

    with pytest.raises(
        KeyError,
        match="No property with name density found in domain medium",
    ):
        material.medium.property("density")
