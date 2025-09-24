# OGSTools 0.7.1 Release Notes

- Works with ogs==6.5.6
- Python: 3.10 - 3.13

## Breaking changes

## API breaking changes

- meshlib.gen_bhe_mesh uses now shapely for meshing geometries see !324:
  - Supports georeferenced modeling in any metric CRS.
  - Flexible model area shapes (no rectangle restriction).
  - Prism mesh works for all cases; structured mesh for well-arranged setups.
  - Limitation: multiple refinement lines in one split surface not yet supported.
  - now output an additional submesh for groundwater downstream
  - how to specify the input, to get the same behavior as before:
    - length and width need to be replaced by model_area
    - Example: `length=150, width=100: model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=150, ymax=100)`
    - dist_box_x and dist_box_y need to be replaced by refinement_area (defined in global coordinate system and not relative to the BHE as before)
      - Example:
        ```
        dist_box_x=10, dist_box_y=5:
        Polygon.from_bounds(
          xmin=min(bhe.x for bhe in bhe_array)-10,
          ymin=min(bhe.y for bhe in bhe_array)- 5,
          xmax=max(bhe.x for bhe in bhe_array)+10,
          ymax=max(bhe.y for bhe in bhe_array)+ 5,
          )
        ```
    - Groundwater.flow_direction needs to be replaced by upstream and downstream
      - Example: `"x": upstream=(179,181), downstream=(359,1)`

## Deprecations

- Planned for `ot.meshes_from_gmsh` (replacement already available: `Meshes.from_gmsh`)

### Examples

## Changes (non API-breaking)

## Bugfixes

- `plot`: Fixed contour plots with very small value ranges (\< 1e-12) to display correctly
- `Project`
  - `run_model` : Fix execution of wrapper commands (e.g. source MKL)
  - Add the missing tag outputs to allow multiple output definitions
- Logparser: fix model_time analysis
- `MeshSeries`:
  - Fixed accessing meshes of extended MeshSeries from PVD files by ensuring timestep_files is updated correctly after extend().
  - Improved scale method to prevent crashes in complex, performance-intensive workflows; new implementation reuses cached meshes for better stability and memory efficiency.
  - fix extract probe for MeshSeries with a single point meshes
- `Mesh`:
  - Fixed crash in Mesh.difference() when meshes contain \_active datasets; computation now handles active field arrays
- `Feflow`:
  - Fixed handling of vector and tensor properties with NaN values (#135): tuples containing only NaNs are now dropped correctly, and tensor values are no longer misclassified

## Features

- Added `MaterialManager` and `MediaSet` to core API and further helper classes: `Phase`, `Medium`, `Material`, `Component`
  - It introduces a modular and schema-driven material handling system for OGS project files.
  - It includes:
    - A YAML-based material database
    - Schema filtering via MaterialList
    - Export to XML (.to_prj()) compatible with OGS
    - Examples for TH2M
  - It is limited to TH2M for now
- Added `Meshes` class
  - `from_simulator` works directly with the OGS mesh in a running simulation
  - `from_gmsh` is `ot.meshes_from_gmsh`
  - `save` performs identify_subdomains, and checks for overwrite
  - `from_msh`
    - Uses the newly introduced `extract_boundaries` for simple extraction of boundary meshes from a 2D domain mesh
  - `from_yaml`
    - Introduces a new tool to generate Gmsh meshes from YAML geometry descriptions.
      - Based on a simple declarative schema (parameters, points, lines, surfaces, groups)
      - Generates .msh files via Gmsh, `meshes_from_yaml`
- Added `plot.contourf_pv` for pyvista plots which work with `Variables`.
- `plot.contourf` now dispatches depending on the value of the new argument `interactive`:
  - None (default): 2D mesh -> matplotlib plot, 3D mesh -> interactive pyvista plot
  - True: always interactive pyvista plot
  - False: 2D mesh -> matplotlib plot, 3D mesh -> pyvista plot screenshot
- Added `to_polar` method for Variables. Useful, to convert stresses to a cylindrical or spherical coordinate system.
- Added `SimulationController` as wrapper for `OGSSimulation` to allow interruption and continuing simulations
- BHE
  - Allow ID notation

## Infrastructure

- Updated requirements!
- Added Binder links to example notebooks for interactive execution.
- Removed EXPERIMENTAL folder -> see Gitlab open issues
- Re-enable testing for Mac
- Faster pipeline execution (10min -> 6min)

## Documentation

- Added tutorials for interactive ogs simulation

### Tests

### Imports

## Maintainer TODOs

### next sub release

### next main release

- deprecation notice for `ot.meshes_from_gmsh`
