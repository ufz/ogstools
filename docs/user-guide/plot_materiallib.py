"""
Materials and Media
===================

.. sectionauthor:: Norbert Grunwald

NB! Works for HEAT_CONDUCTION and TH2M (with Phase Transitions) only so far.

This example shows how to build OpenGeoSys media definitions directly
from a YAML-based **Material Library** using
:class:`~ogstools.materiallib.core.material_manager.MaterialManager`
and :class:`~ogstools.materiallib.core.media.MediaSet`.

The YAML schema defines materials with:

* **name** - unique material identifier
* **properties** - sets of constitutive relations (with type, parameters, scope)
* **scope** - indicates whether a property applies at *phase* or *medium* level

Together with the built-in **process schemas** (e.g. ``TH2M_PT``),
these building blocks allow you to construct full **Media** definitions
including phases and components, and import them into an OGS project XML
via :meth:`~ogstools.ogs6py.Project.set_media`.
"""

# %%
from pathlib import Path
from tempfile import mkdtemp

import ogstools as ot
from ogstools import definitions as defs

model_dir = Path(mkdtemp())

# %% [markdown]
# Example materials
# -----------------
#
# Materials are provided as YAML files in the built-in Material Library.
# For example, here are the definitions of "opalinus_clay" (solid) and "water" (fluid):

# %%
print((Path(defs.MATERIALS_DIR) / "opalinus.yml").read_text(encoding="utf-8"))
print((Path(defs.MATERIALS_DIR) / "water.yml").read_text(encoding="utf-8"))

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
        "material_ids": [0, 3, 4],  # multiple MatIDs grouped under one name
    },
    {
        "subdomain": "buffer",
        "material": "bentonite",
        "material_ids": [1, 2],
    },
]

fluids = {
    "AqueousLiquid": "water",
    "Gas": "carbon_dioxide",
}  # required by TH2M_PT schema

filtered = db.filter(process="TH2M_PT", subdomains=subdomains, fluids=fluids)

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
