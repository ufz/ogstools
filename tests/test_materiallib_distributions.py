# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import]

from ogstools import Project
from ogstools.materiallib.core import material_manager
from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.material_manager import MaterialManager
from ogstools.materiallib.core.media import MediaSet
from ogstools.materiallib.core.property import PropertyAddress


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


def test_property_address_reads_top_level_baseline_and_distribution(
    tmp_path: Path,
) -> None:
    material_path = _write_yaml(
        tmp_path / "porosity.yml",
        {
            "name": "porosity_test",
            "domains": [
                {
                    "domain": "medium",
                    "properties": {
                        "porosity": [
                            {
                                "type": "Constant",
                                "value": {
                                    "value": 0.15,
                                    "distribution": {
                                        "type": "uniform",
                                        "min": 0.1,
                                        "max": 0.2,
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

    address = PropertyAddress(domain="medium", property_name="porosity")
    assert material.baseline_value(address) == 0.15
    assert material.distribution(address) == {
        "type": "uniform",
        "min": 0.1,
        "max": 0.2,
    }


def test_property_address_reads_nested_baseline_and_distribution(
    tmp_path: Path,
) -> None:
    material_path = _write_yaml(
        tmp_path / "nested.yml",
        {
            "name": "nested_test",
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
                                    "distribution": {
                                        "type": "loguniform",
                                        "min": 1.0e7,
                                        "max": 1.0e8,
                                    },
                                },
                                "residual_gas_saturation": 0.01,
                            }
                        ]
                    },
                }
            ],
        },
    )

    material = Material.from_file(material_path)
    assert material is not None

    exponent = PropertyAddress(
        domain="medium",
        property_name="saturation",
        parameter_path=("exponent",),
    )
    assert material.baseline_value(exponent) == 0.2
    assert material.distribution(exponent) == {
        "type": "uniform",
        "min": 0.15,
        "max": 0.3,
    }

    p_b = PropertyAddress(
        domain="medium",
        property_name="saturation",
        parameter_path=("p_b",),
    )
    assert material.baseline_value(p_b) == 4.8e7
    assert material.distribution(p_b) == {
        "type": "loguniform",
        "min": 1.0e7,
        "max": 1.0e8,
    }

    plain_scalar = PropertyAddress(
        domain="medium",
        property_name="saturation",
        parameter_path=("residual_gas_saturation",),
    )
    assert material.baseline_value(plain_scalar) == 0.01
    assert material.distribution(plain_scalar) is None


def test_property_address_uses_domain_and_index_to_select_property_variant(
    tmp_path: Path,
) -> None:
    material_path = _write_yaml(
        tmp_path / "thermal_conductivity.yml",
        {
            "name": "thermal_test",
            "domains": [
                {
                    "domain": "medium",
                    "properties": {
                        "thermal_conductivity": [
                            {
                                "type": "Constant",
                                "value": {
                                    "value": 1.7,
                                    "distribution": {
                                        "type": "uniform",
                                        "min": 1.5,
                                        "max": 1.9,
                                    },
                                },
                            },
                            {
                                "type": "Constant",
                                "value": {
                                    "value": 2.1,
                                    "distribution": {
                                        "type": "uniform",
                                        "min": 2.0,
                                        "max": 2.2,
                                    },
                                },
                            },
                        ]
                    },
                },
                {
                    "domain": "phase",
                    "properties": {
                        "thermal_conductivity": [
                            {"type": "Constant", "value": 5.0}
                        ]
                    },
                },
            ],
        },
    )

    material = Material.from_file(material_path)
    assert material is not None

    medium_first = PropertyAddress(
        domain="medium", property_name="thermal_conductivity", index=0
    )
    medium_second = PropertyAddress(
        domain="medium", property_name="thermal_conductivity", index=1
    )
    phase_first = PropertyAddress(
        domain="phase", property_name="thermal_conductivity", index=0
    )

    assert material.baseline_value(medium_first) == 1.7
    assert material.distribution(medium_first) == {
        "type": "uniform",
        "min": 1.5,
        "max": 1.9,
    }
    assert material.baseline_value(medium_second) == 2.1
    assert material.distribution(medium_second) == {
        "type": "uniform",
        "min": 2.0,
        "max": 2.2,
    }
    assert material.baseline_value(phase_first) == 5.0
    assert material.distribution(phase_first) is None


def test_export_writes_only_baseline_values_without_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_yaml(
        tmp_path / "host.yml",
        {
            "name": "host",
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
                        ],
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
                                "residual_gas_saturation": {
                                    "value": 0.01,
                                    "distribution": {
                                        "type": "uniform",
                                        "min": 0.0,
                                        "max": 0.02,
                                    },
                                },
                                "residual_liquid_saturation": 0.01,
                            }
                        ],
                    },
                }
            ],
        },
    )

    monkeypatch.setitem(
        material_manager.PROCESS_SCHEMAS,
        "distribution_export_dummy",
        {"properties": ["permeability", "saturation"], "phases": []},
    )

    manager = MaterialManager(data_dir=tmp_path)
    filtered = manager.filter(
        process="distribution_export_dummy",
        subdomains=[
            {"subdomain": "region1", "material": "host", "material_ids": [0]}
        ],
        fluids={},
    )
    media = MediaSet(filtered)

    project = Project()
    project.set_media(media)

    prj_path = tmp_path / "distribution_export.prj"
    project.write_input(prj_path)
    xml = prj_path.read_text(encoding="utf-8")

    assert "<name>permeability</name>" in xml
    assert "<value>1e-20</value>" in xml
    assert "<name>saturation</name>" in xml
    assert "<exponent>0.2</exponent>" in xml
    assert "<p_b>48000000.0</p_b>" in xml
    assert "<residual_gas_saturation>0.01</residual_gas_saturation>" in xml
    assert (
        "<residual_liquid_saturation>0.01</residual_liquid_saturation>" in xml
    )
    assert "distribution" not in xml
