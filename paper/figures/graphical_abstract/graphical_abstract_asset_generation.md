# Graphical abstract — asset generation (handoff for a new session)

## Context

We iteratively designed an OGSTools graphical abstract across a design session, previewing it as an HTML Artifact. Current draft: https://claude.ai/code/artifact/943f8add-e44c-4448-8d7c-08fcbf9776a8 (redeployed repeatedly at the same URL — open it to see the latest state).

The layout itself is done. A generation script now exists — `paper/figures/generate_graphical_abstract_assets.py` — and already produces 5 of the real images (see "Done" below), all from **one consistent running example** (the paper's own Liquid Flow 2D case, same as `figure1_boundary_conditions.png`). What's left for **this new session**: extend that script to cover the remaining items, and fold everything into the final `paper/figures/graphical_abstract.svg` (currently a simpler v1 pipeline diagram, not yet updated to match the final design).

## Third-party icon provenance

The notes row (Python/OGSTools exchange + "Compatible / integratable with" icons) uses two
third-party marks embedded directly in the draft HTML, not yet wired through the generation
script (they aren't matplotlib/pyvista figures, just static vector art):

- `paper/figures/assets/numpy_logo.svg` — copied verbatim from NumPy's own official brand
  assets (`branding/logo/logomark/numpylogoicon.svg` in the numpy source distribution).
- The Python logo mark is the Font Awesome Free 6.5.2 "python" brand icon
  ([CC BY 4.0 license](https://fontawesome.com/license/free)), inlined as an SVG path in the
  HTML and recolored to Python's own official brand blue (`#3776AB`), not the OGSTools blue.

## Design decisions to carry over (don't relitigate)

- Three columns, same typographic style (header + stacked groups, label + thumbnail(s) + short caption line, no card borders): **Pre-processing** (Meshes, Project, Model — Model's caption line also carries the execution axes: Serial/MPI/OpenMP/± container), **Simulation Execution** (OpenGeoSys logo, "optional interactive (co-simulation)", "released / custom OGS"), **Post-processing** (MeshSeries, Log). Plain arrows between them, no text on the arrows.
- The whole three-column row sits inside one bordered frame chipped "single simulation".
- To its right (not below — this saves vertical space), a narrower dashed-border panel chipped "combine simulations", built from an abstract visual language: a small blue rectangle = "one simulation". Two groups, stacked vertically: **Difference** (2 boxes → real diff plot below) and **Convergence** (boxes stacked "übereinander" → real convergence-study plot below, currently pending). **There is no "Restart chain" group** — it was tried twice and dropped; don't re-add it.
- Brand colors: blue `#104EB2` (dark-mode `#7FA6EE`), orange `#B5650A` (dark-mode `#E3A03D`), from `docs/_static/ogstools.css`.
- Wordmark reads "OGSTools" (capital T), not the lowercase logo file.
- Logos: **no per-item tool logos** (gmsh/pyvista/pandas/Matplotlib icons were tried and rejected as "too much"). Only the **OpenGeoSys** logo is used, emphasized, for the Simulation Execution node. Tool compatibility is stated as three short plain-text lines instead: "Meshes & MeshSeries are pyvista-compatible", "All plots can be adapted with Matplotlib", "OpenGeoSys command lines are integrated".

## The running example (already established — keep using it)

`paper/figures/generate_graphical_abstract_assets.py:_liquid_flow_simulation()` builds and runs the exact model from the paper's own quickstart example: 2D rectangular domain (`ot.gmsh_tools.rect((8,4), 8, 2)`), pressure BCs (2.9e7 / 3.1e7), transient `LIQUID_FLOW`, 11 time steps. It actually runs OGS (the wheel is installed; `ot.Model.run()` works in this environment). Any new image should come from this same simulation unless there's a good reason not to (e.g. the "combine" section genuinely needs *multiple* runs).

## Done (generated for real this session, not placeholders)

Script: `paper/figures/generate_graphical_abstract_assets.py`. Outputs land in `paper/figures/assets/`.

- `generate_mesh_3d_prism()` → 3D mesh with prism elements (Meshes). Generic demo cuboid via `ot.gmsh_tools.cuboid(mixed_elements=True)`, **not** tied to the (2D) running example — open question below.
- `generate_meshseries_spatial()` → spatial contourf of pressure at the final time step, with 4 points (P1–P4 at x=1/3/5/7, y=2) marked.
- `generate_meshseries_temporal()` → the same 4 points probed over time (`ms.probe(...)` + `ot.plot.line`), guaranteed spatially consistent with the plot above since both come from one `_liquid_flow_simulation()` call.
- `generate_log_convergence()` → `sim.log.plot_convergence()`, regenerated fresh (not reusing the old `figure2_convergence.png`) so it's from the same run as everything else.
- `generate_log_computational_metrics()` → assembly time & linear solver time per step, from `sim.log.time_step()` (a real pandas DataFrame — columns: `output_time`, `step_size`, `time_step_solution_time`, `assembly_time`, `dirichlet_time`, `linear_solver_time`).

## Still open

1. **2D mesh with edges** (Meshes). Currently still reusing `tests/baseline/test_meshes_plot.png` (a real-world Selke Basin mesh with legend — busy/unrelated to the running example). Consider generating a clean, minimal 2D triangulated mesh from the running example's own domain instead (plain, no legend) — add as `generate_mesh_2d_edges()`.

1. **3D mesh** — decide whether the generic demo cuboid from `generate_mesh_3d_prism()` is fine to keep (it's a different, 3D-only concept than the 2D running example, so full consistency isn't really possible here) or should be dropped/simplified.

1. **OpenGeoSys logo** (Simulation Execution). Currently reused from a local checkout: `~/o/wt/6.5.6/web/static/images/OGS-Logo.png` (also copied ad-hoc to the session's scratchpad as `ogs_icon.png`, cropped). This is outside the repo — copy a permanent version into `paper/figures/assets/` (confirm it's fine to redistribute; it's the official OpenGeoSys project logo) and add e.g. `copy_ogs_logo()` or just check in the static file directly.

1. **MeshSeries difference plot** ("combine simulations" → Difference). Currently still reusing `tests/baseline/test_contourf_diff_von_Mises_stress-xlim__2, 5__ylim__-1.1, None_.png` (mechanics example, unrelated). Regenerate using `ogstools.mesh.differences` (`difference`/`difference_pairwise`/`compare` in `ogstools/mesh/differences.py`) comparing two runs of `_liquid_flow_simulation()` (e.g. two mesh resolutions) — add `generate_meshseries_difference()`.

1. **Convergence-study plot** ("combine simulations" → Convergence). **No existing asset — still a "pending" placeholder in the draft.** This is the heavy one: generate via `ogstools.studies.convergence` (`ogstools/studies/convergence/convergence.py`: `grid_convergence`, `richardson_extrapolation`, `plot_convergence_error_evolution`; driver `ogstools/studies/convergence/study.py:run_convergence_study`). Reference scripts: `docs/examples/howto_postprocessing/plot_convergence_study_steady_state_diffusion.py` and `plot_convergence_study_nuclear_decay.py`. Needs multiple resolutions of the running example — budget more time for this one. Add `generate_convergence_study()`.

## Note on the Project XML snippet

Not an image, just literal text in the HTML template — currently the first lines of `ogstools/examples/prj/TM_square.prj`:

```
<OpenGeoSysProject>
    <meshes>
```

Fine as-is; only revisit if a prj matching the running example (`ogstools/examples/prj/SimpleLF.prj`) reads better there.
