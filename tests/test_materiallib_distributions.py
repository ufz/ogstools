# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import]

from ogstools.materiallib.core.material import Material


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_top_level_wrapped_value_is_parsed_into_baseline_and_metadata(
    tmp_path: Path,
) -> None:
    material_path = _write_yaml(
        tmp_path / "permeability.yml",
        {
            "name": "permeability_test",
            "domains": [
                {
                    "domain": "medium",
                    "properties": {
                        "permeability": [
                            {
                                "type": "Constant",
                                "value": {
                                    "value": 1.0e-20,
                                    "unit": "m²",
                                    "distribution": {
                                        "type": "loguniform",
                                        "min": 1.0e-21,
                                        "max": 1.0e-19,
                                    },
                                },
                            }
                        ]
                    },
                }
            ],
        },
    )

    material = Material.from_file(material_path)

    assert material is not None
    permeability = material["permeability"]
    assert permeability.value == 1.0e-20
    assert permeability.extra["unit"] == "m²"
    assert permeability.extra["distribution"] == {
        "type": "loguniform",
        "min": 1.0e-21,
        "max": 1.0e-19,
    }
    assert permeability.extra["domain"] == "medium"


def test_top_level_wrapped_value_roundtrip_preserves_distribution_metadata(
    tmp_path: Path,
) -> None:
    source = _write_yaml(
        tmp_path / "density.yml",
        {
            "name": "density_test",
            "domains": [
                {
                    "domain": "phase",
                    "properties": {
                        "density": [
                            {
                                "type": "Constant",
                                "value": {
                                    "value": 2600,
                                    "unit": "kg/m³",
                                    "distribution": {
                                        "type": "normal",
                                        "mean": 2600,
                                        "stddev": 50,
                                    },
                                },
                            }
                        ]
                    },
                }
            ],
        },
    )

    material = Material.from_file(source)
    assert material is not None

    roundtrip_path = tmp_path / "density_roundtrip.yml"
    material.to_file(roundtrip_path)

    roundtrip_raw = yaml.safe_load(roundtrip_path.read_text(encoding="utf-8"))
    density_entry = roundtrip_raw["domains"][0]["properties"]["density"][0]
    assert density_entry["value"] == {
        "value": 2600,
        "unit": "kg/m³",
        "distribution": {
            "type": "normal",
            "mean": 2600,
            "stddev": 50,
        },
    }


def test_nested_wrapped_parameter_payload_is_preserved_unchanged(
    tmp_path: Path,
) -> None:
    source = _write_yaml(
        tmp_path / "saturation.yml",
        {
            "name": "saturation_test",
            "domains": [
                {
                    "domain": "medium",
                    "properties": {
                        "saturation": [
                            {
                                "type": "SaturationVanGenuchten",
                                "exponent": {
                                    "value": 0.2,
                                    "distribution": {
                                        "type": "uniform",
                                        "min": 0.15,
                                        "max": 0.3,
                                    },
                                },
                                "p_b": {
                                    "value": 4.8e7,
                                    "unit": "Pa",
                                    "distribution": {
                                        "type": "loguniform",
                                        "min": 1.0e7,
                                        "max": 1.0e8,
                                    },
                                },
                            }
                        ]
                    },
                }
            ],
        },
    )

    material = Material.from_file(source)
    assert material is not None

    saturation = material["saturation"]
    assert saturation.value is None
    assert saturation.extra["exponent"] == {
        "value": 0.2,
        "distribution": {
            "type": "uniform",
            "min": 0.15,
            "max": 0.3,
        },
    }
    assert saturation.extra["p_b"] == {
        "value": 4.8e7,
        "unit": "Pa",
        "distribution": {
            "type": "loguniform",
            "min": 1.0e7,
            "max": 1.0e8,
        },
    }

    roundtrip_path = tmp_path / "saturation_roundtrip.yml"
    material.to_file(roundtrip_path)
    roundtrip_raw = yaml.safe_load(roundtrip_path.read_text(encoding="utf-8"))
    saturation_entry = roundtrip_raw["domains"][0]["properties"]["saturation"][
        0
    ]
    assert saturation_entry["exponent"] == {
        "value": 0.2,
        "distribution": {
            "type": "uniform",
            "min": 0.15,
            "max": 0.3,
        },
    }
    assert saturation_entry["p_b"] == {
        "value": 4.8e7,
        "unit": "Pa",
        "distribution": {
            "type": "loguniform",
            "min": 1.0e7,
            "max": 1.0e8,
        },
    }
