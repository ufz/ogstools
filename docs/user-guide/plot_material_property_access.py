"""
Material Property Access
========================

This example shows how to inspect YAML-based material definitions directly
with :class:`~ogstools.materiallib.core.material.Material`.

The focus is on the property-access helpers introduced for nested parameter
payloads:

* :class:`~ogstools.materiallib.core.property.PropertyAddress`
* :meth:`~ogstools.materiallib.core.material.Material.baseline_value`
* :meth:`~ogstools.materiallib.core.material.Material.distribution`

It also shows when a plain ``material["property_name"]`` lookup is sufficient
and when an explicit address is needed.
"""

# %%
import logging

import yaml  # type: ignore[import]

import ogstools as ot
from ogstools.materiallib.core.material import Material
from ogstools.materiallib.core.property import PropertyAddress

work_dir = ot.definitions.temp_dir("materiallib", "property-access")


# %% [markdown]
# Load a built-in material
# ------------------------
#
# ``Material.from_file()`` is the main entry point for loading one material YAML
# definition.

# %%
opalinus_path = ot.definitions.MATERIALS_DIR / "opalinus.yml"
opalinus = Material.from_file(opalinus_path)

print(f"source: {opalinus_path.name}")
print(f"material: {opalinus.name}")
print(f"properties: {', '.join(list(opalinus)[:5])}, ...")
print(f"property count: {len(opalinus.properties)}")


# %% [markdown]
# Simple property inspection
# --------------------------
#
# For simple inspection, direct lookup by property name is often enough.
# This returns the first property occurrence with that name.

# %%
permeability = opalinus["permeability"]
print(permeability)


# %% [markdown]
# Why `PropertyAddress` exists
# ----------------------------
#
# Some properties carry additional metadata such as uncertainty distributions.
# Others contain nested parameter payloads instead of a single scalar value.
# ``PropertyAddress`` is used to point to one concrete property occurrence:
#
# * ``domain``: top-level domain block such as ``medium`` or ``phase``
# * ``property_name``: the property key inside that domain
# * ``index``: which occurrence to select if the same property appears multiple times
# * ``parameter_path``: optional nested path inside the property payload
#
# To keep the example self-contained, we create a small temporary material file
# with one top-level wrapped value and one nested wrapped parameter.

# %%
material_path = work_dir / "property_access_demo.yml"
material_path.write_text(
    yaml.safe_dump(
        {
            "name": "property_access_demo",
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
                                        "min": 0.10,
                                        "max": 0.20,
                                    },
                                },
                            }
                        ],
                        "saturation": [
                            {
                                "type": "SaturationVanGenuchten",
                                "exponent": {
                                    "value": 0.20,
                                    "distribution": {
                                        "type": "uniform",
                                        "min": 0.15,
                                        "max": 0.30,
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
                        ],
                    },
                }
            ],
        },
        sort_keys=False,
    ),
    encoding="utf-8",
)

demo_material = Material.from_file(material_path)
print(f"demo material written to: {material_path.name}")


# %% [markdown]
# Top-level baseline value and distribution
# -----------------------------------------
#
# For a wrapped top-level scalar like ``porosity``, the empty ``parameter_path``
# means "use the property value itself".

# %%
porosity_address = PropertyAddress(
    domain="medium",
    property_name="porosity",
)

print({"baseline": demo_material.baseline_value(porosity_address)})
print({"distribution": demo_material.distribution(porosity_address)})


# %% [markdown]
# Nested parameter access with `parameter_path`
# ---------------------------------------------
#
# For composite properties such as ``SaturationVanGenuchten``, the baseline value
# is stored in nested fields like ``exponent`` or ``p_b``.
#
# ``parameter_path=("exponent",)`` means:
# "start from the addressed property and then look up the nested key
# ``exponent``."

# %%
exponent_address = PropertyAddress(
    domain="medium",
    property_name="saturation",
    parameter_path=("exponent",),
)

p_b_address = PropertyAddress(
    domain="medium",
    property_name="saturation",
    parameter_path=("p_b",),
)

print(
    {
        "exponent baseline": demo_material.baseline_value(exponent_address),
        "exponent distribution": demo_material.distribution(exponent_address),
    }
)
print(
    {
        "p_b baseline": demo_material.baseline_value(p_b_address),
        "p_b distribution": demo_material.distribution(p_b_address),
    }
)


# %% [markdown]
# Repeated properties and `index`
# -------------------------------
#
# If the same property appears multiple times in one domain, ``index`` selects
# the concrete occurrence.

# %%
repeat_path = work_dir / "repeated_property_demo.yml"
repeat_path.write_text(
    yaml.safe_dump(
        {
            "name": "repeated_property_demo",
            "domains": [
                {
                    "domain": "medium",
                    "properties": {
                        "density": [
                            {"type": "Constant", "value": 1.7},
                            {"type": "Constant", "value": 2.1},
                        ]
                    },
                }
            ],
        },
        sort_keys=False,
    ),
    encoding="utf-8",
)

repeat_material = Material.from_file(repeat_path)

first_density = PropertyAddress("medium", "density", index=0)
second_density = PropertyAddress("medium", "density", index=1)

print({"density[0]": repeat_material.baseline_value(first_density)})
print({"density[1]": repeat_material.baseline_value(second_density)})


# %% [markdown]
# Validation timing and failure modes
# -----------------------------------
#
# ``PropertyAddress`` itself is a lightweight selector. The semantic validation
# happens when it is resolved against a concrete ``Material``.
#
# Invalid nested paths raise ``KeyError``; invalid property indices raise
# ``IndexError``.

# %%
try:
    invalid_path = PropertyAddress(
        domain="medium",
        property_name="saturation",
        parameter_path=("does_not_exist",),
    )
    demo_material.baseline_value(invalid_path)
except KeyError as err:
    print(f"{type(err).__name__}: {err}")

try:
    invalid_index = PropertyAddress(
        domain="medium",
        property_name="density",
        index=99,
    )
    repeat_material.baseline_value(invalid_index)
except IndexError as err:
    print(f"{type(err).__name__}: {err}")


# %% [markdown]
# Round-trip export
# -----------------
#
# Materials can be written back to YAML using ``Material.to_file()``.

# %%
roundtrip_path = work_dir / "property_access_roundtrip.yml"
demo_material.to_file(roundtrip_path)
roundtrip_raw = yaml.safe_load(roundtrip_path.read_text(encoding="utf-8"))
print(f"roundtrip file: {roundtrip_path.name}")
print(roundtrip_raw["domains"][0]["properties"]["porosity"][0]["value"])


# %% [markdown]
# Repository access with `MaterialManager`
# ----------------------------------------
#
# For one known file, ``Material.from_file()`` is usually the simplest API.
# If you want to load a repository of material YAML files, use
# ``MaterialManager``.
#
# - ``Material.from_file()`` is intended for one known material file.
# - ``MaterialManager`` is better suited for repository-style loading, where
#   YAML files that are not valid material definitions may be skipped during
#   repository discovery.

# %%
logging.getLogger("ogstools.materiallib.core.material_manager").setLevel(
    logging.WARNING
)
manager = ot.MaterialManager()
water = manager.get_material("water")
print(f"repository material: {water.name}")
print(f"available properties: {', '.join(list(water)[:5])}, ...")
