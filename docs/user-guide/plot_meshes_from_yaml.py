"""
Meshes from YAML
================

.. sectionauthor:: Norbert Grunwald (Helmholtz Centre for Environmental Research GmbH - UFZ)

This example shows how to generate OGS-ready VTU meshes directly from a YAML
geometry description using :meth:`~ogstools.meshlib.meshes.Meshes.from_yaml`.

The YAML schema defines the mesh in terms of five top-level keys:

* **parameters** - scalar values and expressions that can be reused in the geometry
* **points** - coordinates and local mesh sizes
* **lines** - connections between points (straight or circular arcs)
* **surfaces** - closed loops of lines/arcs, optionally with holes
* **groups** - physical groups (domains, boundaries, subregions)

Together these building blocks allow you to describe arbitrary 2D geometries,
assign regions and boundaries, and directly obtain a conforming triangular mesh.

Internally, Gmsh is still used to build the geometry and generate a `.msh` file,
but :meth:`~ogstools.meshlib.meshes.Meshes.from_yaml` converts the result
immediately into a :class:`~ogstools.meshlib.meshes.Meshes` object with VTU
submeshes (domain, boundaries, and groups) that can be used directly in OGS.

"""

# %%
import ogstools as ot
from ogstools.definitions import EXAMPLES_DIR

# %% [markdown]
# Example YAML geometry
# ---------------------
#
# An example geometry is provided in `ogstools/examples/meshlib/meshes_from_yaml/example_hlw.yml`.

# %%
yaml_file = EXAMPLES_DIR / "meshlib/meshes_from_yaml/example_hlw.yml"
yaml_content = yaml_file.read_text(encoding="utf-8")
print(yaml_content)


# %% [markdown]
# Mesh generation
# ---------------

# Using :meth:`~ogstools.meshlib.meshes.Meshes.from_yaml` we create
# a :class:`~ogstools.meshlib.meshes.Meshes` container directly from
# the YAML file.

# Internally, a Gmsh `.msh` file is generated, but it is automatically
# converted into VTU meshes. The returned object already provides
# access to all named meshes (domain, boundaries, groups) and can be
# queried or plotted without further conversion.

# %%
meshes = ot.Meshes.from_yaml(yaml_file)
print(*[f"{name}: {mesh.n_cells=}" for name, mesh in meshes.items()], sep="\n")

# %% [markdown]
# Plot the domain mesh
# --------------------
#
# Here we plot the domain mesh with material IDs shown.

# %%
fig = ot.plot.contourf(
    meshes.domain(), ot.variables.material_id, show_edges=True
)

# %%
# Saving meshes
meshes.save()

# %% [markdown]
# Command line usage
# ------------------

# The same functionality is also available via the command line.
# You can run the tool directly, passing a YAML geometry file and optionally
# an output directory. The tool will generate the intermediate `.msh` file,
# convert it to VTU meshes, and save everything to the output directory.

# .. code-block:: bash

#    yaml2vtu -i example_simple.yml
#    yaml2vtu -i example_simple.yml -o output_dir

# This produces:

# * `output_dir/mesh.msh` - the raw Gmsh mesh
# * `output_dir/mesh_*.vtu` - domain and submeshes in VTU format, ready for OGS
