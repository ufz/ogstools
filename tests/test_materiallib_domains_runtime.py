# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from ogstools.materiallib.core.component import Component
from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.medium import Medium
from ogstools.materiallib.core.phase import Phase
from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS


def _grouped_material(name: str, domains: list[dict]) -> Material:
    return Material(name=name, raw_data={"name": name, "domains": domains})


@pytest.fixture
def grouped_schema(monkeypatch):
    schema = {
        "properties": ["Density"],
        "phases": [
            {
                "type": "AqueousLiquid",
                "properties": ["Density", "Viscosity"],
            }
        ],
    }
    monkeypatch.setitem(PROCESS_SCHEMAS, "grouped_dummy", schema)
    return "grouped_dummy"


@pytest.fixture
def grouped_component_schema(monkeypatch):
    schema = {
        "properties": [],
        "phases": [
            {
                "type": "AqueousLiquid",
                "properties": [],
                "components": {"Solvent": ["MolarMass"], "Solute": []},
            }
        ],
    }
    monkeypatch.setitem(PROCESS_SCHEMAS, "grouped_component_dummy", schema)
    return "grouped_component_dummy"


def test_medium_loads_only_medium_domain_properties(
    grouped_schema: str,
) -> None:
    solid = _grouped_material(
        "rock",
        [
            {
                "domain": "medium",
                "properties": {
                    "Density": [{"type": "Constant", "value": 2400}]
                },
            },
            {
                "domain": "phase",
                "properties": {"Density": [{"type": "Constant", "value": 999}]},
            },
        ],
    )
    fluid = _grouped_material(
        "water",
        [
            {
                "domain": "phase",
                "properties": {
                    "Density": [{"type": "Constant", "value": 999}],
                    "Viscosity": [{"type": "Constant", "value": 1.0}],
                },
            }
        ],
    )

    medium = Medium(
        material_id=1,
        material=solid,
        name="region1",
        fluids={"AqueousLiquid": fluid},
        process=grouped_schema,
    )

    assert [
        prop.value for prop in medium.properties if prop.name == "Density"
    ] == [2400]
    assert medium.properties[0].extra["domain"] == "medium"


def test_phase_loads_only_phase_domain_properties(grouped_schema: str) -> None:
    fluid = _grouped_material(
        "water",
        [
            {
                "domain": "medium",
                "properties": {
                    "Density": [{"type": "Constant", "value": 2400}]
                },
            },
            {
                "domain": "phase",
                "properties": {
                    "Density": [{"type": "Constant", "value": 999}],
                    "Viscosity": [{"type": "Constant", "value": 1.0}],
                },
            },
        ],
    )

    phase = Phase(
        phase_type="AqueousLiquid",
        liquid_material=fluid,
        process=grouped_schema,
    )

    density = next(prop for prop in phase.properties if prop.name == "Density")
    assert density.value == 999
    assert density.extra["domain"] == "phase"


def test_component_loads_only_component_domain_properties(
    grouped_component_schema: str,
) -> None:
    component_material = _grouped_material(
        "water",
        [
            {
                "domain": "phase",
                "properties": {
                    "MolarMass": [{"type": "Constant", "value": 18.0}]
                },
            },
            {
                "domain": "component",
                "properties": {
                    "MolarMass": [{"type": "Constant", "value": 18.0}]
                },
            },
        ],
    )

    component = Component(
        material=component_material,
        phase_type="AqueousLiquid",
        role="Solvent",
        process=grouped_component_schema,
        diffusion_coefficient=0.0,
    )

    assert [prop.name for prop in component.properties] == ["MolarMass"]
    assert component.properties[0].extra["domain"] == "component"
