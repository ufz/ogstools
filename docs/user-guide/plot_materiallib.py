"""
Materials and Media
===================

NB! This example demonstrates the stable `HEAT_CONDUCTION` path.

This example shows how to build OpenGeoSys media definitions directly
from a YAML-based **Material Library** using
:class:`~ogstools.materiallib.core.material_manager.MaterialManager`
and :class:`~ogstools.materiallib.core.media.MediaSet`.

The YAML schema defines materials with:

* **name** - unique material identifier
* **domains** - grouped `medium` / `phase` / `component` property blocks
* **properties** - sets of constitutive relations (with type and parameters)

Together with the built-in **process schemas** (e.g. ``HEAT_CONDUCTION``),
these building blocks allow you to construct full **Media** definitions
including phases and components, and import them into an OGS project XML
via :meth:`~ogstools.Project.set_media`.
"""

# %%


import ogstools as ot

model_dir = ot.definitions.temp_dir("materiallib", "user-guide")

# %% [markdown]
# Example materials
# -----------------
#
# Materials are provided as YAML files in the built-in Material Library.
# For example, here are the definitions of "opalinus_clay" (solid) and "water" (fluid):

# %%
print(
    (ot.definitions.MATERIALS_DIR / "opalinus.yml").read_text(encoding="utf-8")
)
print((ot.definitions.MATERIALS_DIR / "water.yml").read_text(encoding="utf-8"))

# %% [markdown]
# Media creation
# --------------
#
# We build a `MaterialManager` from the built-in library, filter it with a schema,
# and construct a `MediaSet` object.
#
# - ``subdomain``: the subdomain name (string, one per entry)
# - ``material``: must match the ``name`` in the YAML file
# - ``material_ids``: list of integers corresponding to the MatIDs in the mesh
#   (allows grouping several mesh regions under one subdomain name)

# %%
db = ot.MaterialManager()

subdomains = [
    {
        "subdomain": "host_rock",
        "material": "opalinus_clay",
        "material_ids": [0, 1, 2, 3, 4],
    }
]

filtered = db.filter(
    process="HEAT_CONDUCTION", subdomains=subdomains, fluids={}
)

media = ot.MediaSet(filtered)

# %% [markdown]
# Export to OGS Project XML
# -------------------------
#
# The `MediaSet` is imported into an OGS Project instance
# via ``Project.set_media()``.

# %%
prj = ot.Project()
prj.set_media(media)

xml_file = model_dir / "material_test.prj"
prj.write_input(xml_file)
print(xml_file.read_text())
