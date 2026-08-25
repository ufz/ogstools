# OGSTools 0.8.2 Release Notes

Python 3.12-3.14
OGS 6.5.8

# Highlights

- **OGSTools paper submitted to the Journal of Open Source Software** (`paper/paper.md`, `paper/paper.bib`) ([!321](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/321))

# Breaking changes

## API breaking changes

- ogs6py: BHE `flow_and_temperature_control` types were renamed/consolidated to match OGS 6.5.8.dev (e.g. `FixedPowerConstantFlow` and the various curve-based variants are replaced by `InflowTemperature`, `Power`, `BuildingPower`), and `flow_rate` is now supplied as a Parameter for every type. ([!455](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/455))

# Changes (non API-breaking)

## Bugfixes

- fixed sorting in Meshes.from_files (if key was not in the beginning, it didn't sort correctly) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- allow values for group based parameters in `Project.parameters` (allows to set vectorial or anisotropic group values. Previously, only allowed scalars) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- fixed `Project.process_variables.add_bc` and `Project.process_variables.add_st` to allow all different kinds of types. Previously, some
  checks caused unintentional errors. ([!471](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/471))
- fixed `Meshseries.probe` with linear interpolation and cell_data yielding nan at the mesh boundary as cell centers are used for interpolation. Now used nearest neighbor as a fallback. ([!473](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/473))
- fixed reading xdmf data where some data field are only present in the first timestep. Now they will also be read for later timesteps. ([!473](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/473))
- fixed `model.plot_constraints` unintentionally deleting meshes from `model` as a side effect; it now operates on a copy. ([!463](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/463))
- fixed `plot.contourf` raising `unexpected keyword argument 'mask'`. ([!467](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/467))
- fixed the data aspect ratio in `plot.contourf` when `xlim`/`ylim` are given. ([!470](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/470))
- fixed the error message raised by `StorageBase._pre_save` to reference the correct `overwrite` argument. ([!479](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/479))
- relaxed datatype validation for point coordinates and `MaterialIDs` so `float32`/`int32`/`uint32` variants are also accepted, not just the exact reference dtype. ([!475](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/475))
- `ot.mesh.difference` no longer drops the `spatial_unit` attribute of the resulting mesh. ([!468](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/468))
- storage paths/filenames may now contain `~`. ([!461](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/461))

## Features

- plot
  - added `xlim` and `ylim` as arguments for `contourf` ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - improved `meshes.plot` ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - fixed line ordering by trying to sort by cell adjacency ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - `contourf`: corner elements were missing if none of their points were in the selected x- or y-limits; now they are checked explicitly. ([!482](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/482))
- meshes
  - `Meshes.from_mesh` supports quadratic mesh ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - `Meshes` now support `output_names`, saved/restored via `Meshes.save()`/`from_file()`/`from_folder()` and used by `Meshes.plot` and `Model.plot_constraints` for labeling ([!484](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/484))
- mesh
  - `filepath` is now attached to mesh upon reading (with `ot.mesh.read`) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
  - added `ogstools.mesh.utils.pv_set_attr` helper, replacing direct use of the deprecated `pyvista.set_new_attribute` ([!474](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/474))
  - `node_reordering` (and the `NodeReordering` CLI call it wraps) gained a `log` parameter to control/silence its log output ([!488](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/488))
- Project
  - `BuildTree.populate_tree` now can handle dicts as the element text
    (in this case the key and value pairs will create corresponding subelements) ([!472](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/472))
  - Added `added deactivate_subdomain` to `Project.process_variables` ([!472](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/472))
- variables
  - dataname of a variable is now also looked for in the attributes of mesh (e.g. points) or meshseries (e.g. timevalues) ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))
  - functions of nested variables are now stored in list of Functions (instead of nested lambdas) which allows to retrospectively modify arguments and parameters ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))
  - arguments of stored functions can be strings which represent the names of data inside a mesh or meshseries to allow for variables depending on different input data ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))
  - outputname can now explicitly be an empty string ([!487](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/487))

## Infrastructure

- added lychee make command to check for broken links in the docs ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- CI no longer pins a specific Python version, just `3.x`. ([!480](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/480))
- gallery hash checks now exclude plots whose figures embed computational metrics (e.g. solver time), since those vary between runs; pre-commit no longer reformats `paper.md`, and codespell now also skips `*.svg`/`*.html`. ([!321](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/321))

## Documentation

- added CONTRIBUTING.md and linked it from the README, docs nav, and docs site; split developer and maintainer guides; fixed install instructions per the JOSS review. ([!480](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/480))
- fixed broken weblinks in documentation ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))
- submitted the OGSTools paper to the Journal of Open Source Software (`paper/paper.md`, `paper/paper.bib`); added a graphical abstract linked from the README. ([!321](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/321))

### New Examples

- new meshes example (Selke) ([!465](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/465))

### Updated Examples

- reworked simple simulation example, added missing `print()`/`plt.show()` calls across gallery examples, and other example polish from the JOSS paper review. ([!480](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/480))
- BHE example additionally shows how to access temperatures via the `componenents_bhe` structure. ([!480](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/480))
- quickstart examples now plot convergence behaviour as a heatmap of the relative error per iteration/timestep (`SimulationLog.plot_convergence(metric="dx_x")`) instead of clock time per solver phase. ([!321](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/321))

### Tests

- added end-to-end example-project test coverage and shared property-type-registry tests for materiallib. ([!464](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/464))

### Beta testing features

- materiallib YAML schema tightened: a property entry must now be a single mapping (lists of entries are no longer accepted), the `type` key is required, and unsupported keys are rejected. Property parameters are validated against a shared type registry. ([!464](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/464))
- materiallib YAML schema: materials must now group their properties under top-level `domains` (`medium`/`phase`/`component`) instead of a flat `properties` block. Bundled example materials were migrated accordingly. ([!451](https://gitlab.opengeosys.org/ogs/tools/ogstools/-/merge_requests/451))
