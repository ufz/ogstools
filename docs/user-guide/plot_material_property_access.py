"""
Material Property Access
========================

This example shows the structured property access introduced for
``materiallib`` materials:

* :meth:`~ogstools.materiallib.core.material.Material.from_file`
* ``material.medium.property(name)``
* ``material.phase.property(name)``
* ``material.component.property(name)``
* ``material_property.parameter(name)``

The API mirrors the grouped YAML structure of material definitions and avoids
manual searches through flat property lists.

The example also shows how wrapped parameter values with uncertainty metadata
are exposed as
:class:`~ogstools.materiallib.core.property.ParameterValue` objects.
"""

# %%

from pathlib import Path

import yaml  # type: ignore[import]

import ogstools as ot
from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.property import ParameterValue

model_dir = ot.definitions.temp_dir("materiallib", "user-guide")

# %% [markdown]
# Example material
# ----------------
#
# We create one small grouped-domain material with:
#
# * two medium properties,
# * a medium parameter with wrapped uncertainty metadata,
# * a phase property with a plain scalar parameter,
# * a component property with a plain scalar parameter.

# %%
porosity = {
    "type": "Constant",
    "value": {
        "base_value": 0.15,
        "distribution": {
            "type": "uniform",
            "lower": 0.10,
            "upper": 0.20,
        },
    },
}

storage = {"type": "Constant", "value": 1.2e-10}

density = {"type": "Constant", "value": 999.1}

molar_mass = {"type": "Constant", "value": 18.01528}

medium_properties = {
    "porosity": porosity,
    "storage": storage,
}

phase_properties = {
    "density": density,
}

component_properties = {
    "molar_mass": molar_mass,
}

material_data = {
    "name": "demo_material",
    "domains": [
        {"domain": "medium", "properties": medium_properties},
        {"domain": "phase", "properties": phase_properties},
        {"domain": "component", "properties": component_properties},
    ],
}

material_file = Path(model_dir) / "demo_material.yml"
material_file.write_text(
    yaml.safe_dump(material_data, sort_keys=False), encoding="utf-8"
)

print(material_file.read_text(encoding="utf-8"))

# %% [markdown]
# Load the material
# -----------------
#
# The material is loaded with :meth:`Material.from_file`.

# %%
material = Material.from_file(material_file)
print(material)

# %% [markdown]
# Domain-based property navigation
# --------------------------------
#
# The new structured navigation follows the top-level material domains:
# ``medium``, ``phase``, and ``component``. This makes grouped YAML domains
# directly accessible in Python.

# %%
medium_property = material.medium.property("porosity")
storage_property = material.medium.property("storage")
phase_property = material.phase.property("density")
component_property = material.component.property("molar_mass")

print("Medium porosity:", medium_property)
print("Medium storage:", storage_property)
print("Phase density:", phase_property)
print("Component molar mass:", component_property)

# %% [markdown]
# Parameter access
# ----------------
#
# Parameters can now be accessed explicitly from a property object.
#
# At the moment, :meth:`material_property.parameter(name)
# <ogstools.materiallib.core.property.MaterialProperty.parameter>` returns
# either a plain scalar or a
# :class:`~ogstools.materiallib.core.property.ParameterValue`, depending on
# the YAML input form. This transitional behavior is intentional and is
# expected to be normalized in a later merge request.

# %%
porosity_value = medium_property.parameter("value")
storage_value = storage_property.parameter("value")
phase_density = phase_property.parameter("value")
component_molar_mass = component_property.parameter("value")

print("Porosity parameter:", porosity_value)
print("Storage parameter:", storage_value)
print("Phase density parameter:", phase_density)
print("Component molar mass parameter:", component_molar_mass)

# %% [markdown]
# Wrapped uncertainty values
# --------------------------
#
# Wrapped parameter values are returned as
# :class:`~ogstools.materiallib.core.property.ParameterValue` and expose both
# the deterministic value used for export and the optional distribution object.

# %%
assert isinstance(porosity_value, ParameterValue)

print("Base value:", porosity_value.base_value)

if porosity_value.distribution is not None:
    print("Distribution type:", type(porosity_value.distribution).__name__)
    print("Lower bound:", porosity_value.distribution.lower)
    print("Upper bound:", porosity_value.distribution.upper)
