# TODOs for MPyL-API Development

- \[x\] Some parameters (thermal conductivity) are required as phase properties AND medium properties. Usually, as medium property we utilize the 'EffectiveThermalConductivityPorosityMixing' in case of thermal cond.
- \[ \] Raise error when properties appear multiple times in the material database
  → User must select a specific version when loading
- \[x\] Implement media validator
  → Checks completeness and validity of properties, phases, and components
- \[ \] Collect all missing properties in a list (no immediate failure)
  → Prepare `ErrorCollector` class (e.g., `AggregatedPropertyError`)
- \[x\] Automatically create media from `MaterialList` (`create_all_media(...)`)
- \[ \] Store `Medium` objects inside the `prj` object
- \[ \] Unit tests for `Phase`, `Component`, and `Medium`
- \[ \] Test case: full material flow including XML generation
- \[ \] Test case: failure when required properties are missing
- \[ \] Link `medium.name` and `material_id` during export (for clear mapping)
- \[ \] Summary function: `.summary()` or `__repr__()` as a compact overview
- \[ \] Group media by subdomain name or material type
- \[ \] Access individual properties, phases, or components from the `Media` object
- \[ \] Export individual properties, phases, or components to XML
- \[ \] Reverse-parsing: reconstruct media from existing `.prj` files
