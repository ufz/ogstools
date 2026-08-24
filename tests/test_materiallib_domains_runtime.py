# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import pytest

from ogstools.materiallib.core.component import Component
from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.medium import Medium
from ogstools.materiallib.core.phase import Phase
from ogstools.materiallib.core.property import ParameterValue
from ogstools.materiallib.schema.process_schema import PROCESS_SCHEMAS

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[1] / "ogstools/examples/materiallib"
)


def _load_example_material(filename: str) -> Material:
    return Material.from_file(EXAMPLES_DIR / filename)


@pytest.fixture
def grouped_schema(monkeypatch):
    schema = {
        "properties": ["density"],
        "phases": [
            {
                "type": "AqueousLiquid",
                "properties": ["density", "viscosity"],
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
                "components": {"Solvent": ["molar_mass"], "Solute": []},
            }
        ],
    }
    monkeypatch.setitem(PROCESS_SCHEMAS, "grouped_component_dummy", schema)
    return "grouped_component_dummy"


def test_medium_loads_only_medium_domain_properties(
    grouped_schema: str,
) -> None:
    solid = _load_example_material("distributed_demo.yml")
    fluid = _load_example_material("water.yml")

    medium = Medium(
        material_id=1,
        material=solid,
        name="region1",
        fluids={"AqueousLiquid": fluid},
        process=grouped_schema,
    )

    assert [
        prop.parameters["value"]
        for prop in medium.properties
        if prop.name == "density"
    ] == [ParameterValue(base_value=2700)]
    assert medium.properties[0].extra["domain"] == "medium"


def test_phase_loads_only_phase_domain_properties(grouped_schema: str) -> None:
    fluid = _load_example_material("water.yml")

    phase = Phase(
        phase_type="AqueousLiquid",
        liquid_material=fluid,
        process=grouped_schema,
    )

    density = next(prop for prop in phase.properties if prop.name == "density")
    assert density.parameters["value"] == ParameterValue(base_value=1000)
    assert density.extra["domain"] == "phase"


def test_component_loads_only_component_domain_properties(
    grouped_component_schema: str,
) -> None:
    component_material = _load_example_material("water.yml")

    component = Component(
        material=component_material,
        phase_type="AqueousLiquid",
        role="Solvent",
        process=grouped_component_schema,
        diffusion_coefficient=0.0,
    )

    assert [prop.name for prop in component.properties] == ["molar_mass"]
    assert component.properties[0].extra["domain"] == "component"
