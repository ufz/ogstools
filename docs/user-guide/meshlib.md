# meshlib

## Overview

`ogstools.mesh` is a collection of utilities built around [PyVista](https://docs.pyvista.org)
`UnstructuredGrid` objects — so the entire PyVista
ecosystem (visualization, slicing, probing, format export) works out of the box. `ogstools.mesh`
adds only what PyVista doesn't know about: OGS data conventions, validation, integration point
data, and layer-based mesh creation.

For creating meshes from scratch, `ogstools.gmsh_tools` wraps the
[Gmsh Python API](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-API) and exposes it with
OGS-ready defaults (material IDs, boundary groups).

API references: [](../reference/ogstools.mesh.rst) · [](../reference/ogstools.gmsh_tools.rst)

## What makes an OGS mesh special

An OGS mesh is presented as `pv.UnstructuredGrid` that carries several OGS-specific data arrays:

| Name | Location | dtype | Purpose |
| ---- | -------- | ----- | ------- |
| `MaterialIDs` | `cell_data` | `int32` | Labels every element with a material zone; used by OGS to apply material parameters and boundary conditions |
| `bulk_element_ids` | `cell_data` | `uint64` | On boundary/sub-meshes: links back to the element in the parent bulk mesh |
| `bulk_node_ids` | `point_data` | `uint64` | On boundary/sub-meshes: links back to the node in the parent bulk mesh |
| integration point arrays | `field_data` | `float64` | Stress, strain, or other constitutive quantities at Gauss points inside each element |
| `IntegrationPointMetaData` | `field_data` | JSON bytes | Describes each IP array: integration order and number of components |

## Creating meshes

### Simple geometries — `ogstools.gmsh_tools`

`rect()` and `cuboid()` wrap the [Gmsh Python API](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-API)
to produce structured or unstructured 2D/3D meshes with `MaterialIDs` per layer and named boundary
physical groups (`"top"`, `"bottom"`, `"left"`, `"right"`) ready for OGS boundary conditions.
`remesh_with_triangles()` converts an existing quad mesh to triangles.

API: [](../reference/ogstools.gmsh_tools.meshing.rst)

- [](../auto_examples/howto_preprocessing/plot_gen_bhe_mesh.rst)
- [](../auto_examples/howto_preprocessing/plot_remeshing.rst)

### Layered geological domains — `ogstools.mesh.create`

`Surface` → `Layer` → `LayerSet` → `.to_region_prism()` / `.to_region_voxel()` /
`.to_region_tetrahedron()`. Surfaces can be [PyVista](https://docs.pyvista.org) meshes or VTU
files. `MaterialIDs` are propagated automatically from each `Surface`.

API: [](../reference/ogstools.mesh.create.rst) ·
[](../reference/ogstools.mesh.create.boundary_set.rst) (`LayerSet`) ·
[](../reference/ogstools.mesh.create.boundary.rst) (`Layer`) ·
[](../reference/ogstools.mesh.create.boundary_subset.rst) (`Surface`, `Gaussian2D`)

- [](../auto_examples/howto_preprocessing/plot_meshlib_pyvista_input.rst)
- [](../auto_examples/howto_preprocessing/plot_meshlib_vtu_input.rst)

## Reading, validating, and saving

`read` wraps [`pv.read()`](https://docs.pyvista.org/api/utilities/_autosummary/pyvista.read.html)
and returns a `pv.UnstructuredGrid`. `save` additionally strips NaN arrays and syncs
`IntegrationPointMetaData`. `validate` calls the OGS
[checkMesh](https://www.opengeosys.org/stable/docs/tools/meshing/checkmesh/) tool; `node_reordering`
fixes element node ordering for OGS6 compliance.

API: [](../reference/ogstools.mesh.file_io.rst) · [](../reference/ogstools.mesh.utils.rst)

- [](../auto_examples/howto_preprocessing/plot_initial_properties_and_variables.rst)
- [](../auto_examples/howto_preprocessing/plot_extract_boundaries.rst)

## Integration point data

After a simulation, material constitutive quantities (stress and strain tensors, damage variables,
…) are stored at Gauss integration points — one value per integration point per element, not per
node. OGS encodes this in `mesh.field_data` as flat arrays plus a JSON metadata block
(`IntegrationPointMetaData`) that records the integration order and component count for each array.

`IPdata` provides dict-like access to these arrays. `to_ip_mesh()` and `to_ip_point_cloud()`
build visualizable meshes by tessellating each element around its integration points, making the
data explorable in [PyVista](https://docs.pyvista.org) or ParaView.

API: [](../reference/ogstools.mesh.ip_data.rst) (`IPdata`, `ip_metadata`) ·
[](../reference/ogstools.mesh.ip_mesh.rst) (`to_ip_mesh`, `to_ip_point_cloud`, `ip_data_threshold`)

- [](../auto_examples/howto_postprocessing/plot_ipdata.rst)
- [](../auto_examples/howto_preprocessing/plot_modify_integration_point_data.rst)

## Mesh comparison and geometric utilities

- **`difference(base, subtract)`** — subtracts data arrays between two meshes element-by-element;
  resamples automatically when topologies differ.
- **`depth(mesh, top_mesh)`** — interpolates and returns a depth-below-surface scalar from a reference
  surface mesh.
- **`from_simulator(simulation, ...)`** — extracts a live mesh from a running OGS co-simulation
  via the [OGS Python bindings](https://www.opengeosys.org/stable/docs/devguide/advanced/python-wheel/).

API: [](../reference/ogstools.mesh.differences.rst) ·
[](../reference/ogstools.mesh.geo.rst) ·
[](../reference/ogstools.mesh.cosim.rst)

- [](../auto_examples/howto_postprocessing/plot_calculate_diff.rst)
- [](../auto_examples/howto_simulation/plot_200_ogs_interactive_meshes_from_simulator.rst)
- [](../auto_examples/howto_simulation/plot_250_ogs_interactive_mesh_native.rst)
