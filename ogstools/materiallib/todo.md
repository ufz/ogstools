# TODOs for MPyL-API Development

- \[x\] Some parameters (thermal conductivity) are required as phase properties AND medium properties.
  → Medium scope handled separately from phase scope.
- \[ \] Raise error when properties appear multiple times in the material database
  → User must select a specific version when loading.
- \[x\] Implement media validator
  → Checks completeness and validity of properties, phases, and components.
- \[ \] Collect all missing properties in a list (no immediate failure)
  → Prepare `ErrorCollector` class (e.g., `AggregatedPropertyError`).
- \[x\] Replace `MaterialDB` with `MaterialManager` to clarify role.
- \[x\] Build `MediaSet` directly from `MaterialManager.filter` (no `MaterialList` needed).
- \[ \] Store `Medium` objects inside the `Project` object for direct PRJ manipulation.
- \[x\] Unit tests for `Phase`, `Component`, and `Medium` (incl. validation, `to_prj`).
- \[x\] Test case: full material flow including XML generation.
- \[x\] Test case: failure when required properties are missing (raises `ValueError`).
- \[x\] Link `medium.name` and `material_ids` during export (mapping implemented).
- \[ \] Summary function: `.summary()` or `__repr__()` as a compact overview across MediaSet.
- \[ \] Group media by subdomain name or material type.
- \[ \] Access individual properties, phases, or components from the `MediaSet` object.
- \[ \] Export individual properties, phases, or components to XML.
- \[ \] Reverse-parsing: reconstruct media from existing `.prj` files.
- \[x\] Add system tests with real OGS runs (TH2M_PT benchmark, etc.).
- \[ \] Improve error messages (e.g., show available roles/components when mismatch).
- \[ \] Ensure diffusion handling consistent (binary coefficients, correct solvent/solute roles).
