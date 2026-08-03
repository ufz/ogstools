# Graphical abstract — requirements (final decisions)

Consolidated from an iterative design session. Where the user changed their mind mid-session,
only the **final** decision is listed here — no history of superseded requests.

Committed file: `paper/figures/graphical_abstract/graphical_abstract.html` (v38 of the design
session). Live draft (may drift out of sync with the committed file over time — see
`graphical_abstract_manual_editing.md` in this folder for how the two relate):
https://claude.ai/code/artifact/943f8add-e44c-4448-8d7c-08fcbf9776a8

## Structure / layout

- Three stacked sections, full page width and equal width, in this order top to bottom:
  1. **Single simulation** (chip: "single simulation")
  1. **Multiple simulations** (chip: "multiple simulations" — renamed from "combine
     simulations")
  1. **Application** (chip: "application")
- All three sections share the **same style**: solid blue border, blue chip text/border (no more
  per-section color/style distinction — the earlier dashed-orange / black-border variants were
  dropped).
- Within "single simulation": three sub-boxes side by side — **Pre-processing** → **Simulation
  Execution** → **Post-processing** — connected by plain enlarged blue `→` arrows, each sub-box
  individually bordered. All three sub-boxes (and the arrows between them) use `flex: 1` so they
  evenly fill the section's full width, matching the width of Multiple-simulations/Application
  below.
  - Pre-processing contains: Meshes, Project, Model (in that order).
  - Simulation Execution contains, in order: **Simulation** (THMC tetrahedron icon + OGS logo,
    same fixed height for both + description text), **Execution** (Serial · MPI · OpenMP · ±
    Container — one line, capital "Container"), **Co-Simulation / interactive stepping** (loop
    icon + two-line label only — no separate "Other simulation code / mesh manipulation" box),
    **Monitor** (icon + "Live simulation progress" caption).
  - Post-processing contains: MeshSeries (spatial + temporal plots, now stretched to fill the
    column width evenly instead of a fixed pixel width), Log (convergence + computational-metrics
    plots, same stretched treatment).
- Within "multiple simulations": three equal-width bordered cards side by side — **Compare**
  (renamed from "Difference"), **Studies (Multiple Simulations)**, **Chain** (re-added after
  being dropped and re-requested). No real rendered plots inside these cards (see Content section
  below) — icon + caption only.
- Within "application": four equal-width bordered cards side by side — **Workflow
  orchestration**, **Educational notebooks**, **OGS Benchmarks**, **OGS Test suite**.
- Header: a single line reading "**OGS**Tools – A Python library for Open**GeoSys**" (no separate
  bordered tagline box below it anymore — merged into one line, same font size/weight
  throughout), with extra vertical gap below it before the first section.

## Design / styling

- Shared CSS base classes used instead of duplicating styles: `.frame` (position/radius/padding/
  border, now identical across all three top-level sections) and `.box` (border/radius/
  background shared by every inner card: the Pre/Exec/Post columns, the Application cards, and
  the Multiple-simulations cards — `.app-group`/`.combine-group` are one shared CSS rule under
  two class names).
- All box/section headers (`Meshes`, `Project`, `Model`, `Workflow orchestration`, `Studies (Multiple Simulations)`, `Compare`, etc., including the three column titles `Pre-processing` /
  `Simulation Execution` / `Post-processing`) are colored **blue**, in **normal case** (not
  all-caps).
- Only the three outermost chip labels (`single simulation`, `multiple simulations`,
  `application`) and the legend text (`= SINGLE SIMULATION`) stay **all-caps**.
- All body/label/caption text sizes bumped ~20% over the original draft sizes.
- "Co-Simulation" / "interactive stepping" label: same font size as other captions (e.g.
  "Monitor"'s caption), on two separate lines, normal capitalization (not forced uppercase).
- "Single simulation" symbol (the small blue rectangle used throughout — legend, Compare, Studies,
  Chain) always renders at the **same fixed size** everywhere — no per-context size overrides.
- Wordmark colors ("OGS" blue / "Tools..." normal text color / "GeoSys" blue) are the *only*
  color-coding left in the header; never recolor "OGS"/"GeoSys" away from blue.
- Compare card icon: two blue boxes with a **minus sign** between them (box − box), not just two
  boxes side by side.
- Separator style consistency: use middot (`·`) between short terms everywhere (e.g. "Serial ·
  MPI · OpenMP · ± Container", "Steady state · Transient · Restart") — not literal asterisks or
  arrows for this kind of list.
- Model boundary-conditions plot: the two boundary-highlight lines must be clearly visible
  against the mesh fill (white halo behind each colored line) — left boundary **cyan**, right
  boundary **orange** (the original brand orange — the only place orange still appears at all,
  since it's baked into the generated plot image, not a CSS color).
- Any trademarked logo (Python, OpenGeoSys, gmsh, pyvista, etc.) keeps its **own official
  colors** — never recolor a third-party logo into OGSTools' palette. (Exception: purely
  decorative in-house icons — e.g. the exchange-arrow glyph, the monitor/benchmark/notebook
  icons — may use OGSTools blue since they aren't anyone's trademark.)
- THMC tetrahedron icon and OGS logo (Simulation Execution) render at the same fixed height.

## Wording / captions

- Pre-processing → Model: caption reads "Constraints plot" under the image.
- Simulation Execution → Simulation: "Finite-element simulations for porous media in
  geosciences and environmental applications."
- Simulation Execution → Execution: "Serial · MPI · OpenMP · ± Container" (capital "Container",
  must fit one line).
- Simulation Execution → Monitor: "Live simulation progress" (not "Live monitoring · watch
  simulation progress").
- Application → Workflow orchestration: "Implemented with workflow management systems –
  Snakemake or AiiDA" (avoid repeating the word "workflow" twice in the same box).
- Application → OGS Benchmarks: "Compare against analytical & reference FEM solutions" — the
  word "benchmark" here means solution-accuracy validation against analytical/reference FEM
  results, **not** performance/timing benchmarking; caption must use "Compare", not "Validate".
- Multiple simulations → Compare (renamed from "Difference") caption: "MeshSeries difference"
  (it is a pressure-field difference, not von Mises stress — don't reintroduce that).
- Multiple simulations → Studies card: labeled "Studies (Multiple Simulations)", single caption
  "Mesh convergence analysis" underneath (the extra "Absolute error (log-log, max)" caption line
  was removed as redundant).
- Multiple simulations → Chain caption: "Steady state · Transient · Restart".
- Notes line (bottom-right, single line, left-aligned): "Python [exchange icon] OGSTools", then
  "Compatible / integratable with" followed by icon (+ short text where noted): gmsh (+"gmsh"),
  pyvista (icon only), Matplotlib (icon only), pandas (+"pandas"), a command-line icon +"OGS
  command line tools", numpy (+"numpy").

## Material / icon usage (provenance)

- **Python logo**: Font Awesome Free 6.5.2 "python" brand icon (CC BY 4.0), inlined as SVG path,
  split diagonally into the two official Python brand colors (`#3776AB` blue / `#FFD43B`
  yellow) — not monochrome, not OGSTools-colored.
- **NumPy logo**: NumPy's own official small logomark, copied into
  `paper/figures/assets/numpy_logo.svg` for provenance (not left as an untracked local-cache
  reference).
- **OpenGeoSys logo**: copied into the repo's own assets (`copy_ogs_logo()`), kept in its
  original colors, not redistributed from an external checkout path.
- **THMC tetrahedron icon** (Simulation Execution): OpenGeoSys' own process-coupling
  illustration, fetched from `opengeosys.org` (`coupling-icons/t-tet.svg`), rasterized with a
  white background for cross-theme legibility, stored at
  `paper/figures/assets/thmc_tetrahedron.png`.
- gmsh / pyvista / Matplotlib / pandas icons: existing small brand-colored icons already used in
  the notes line (kept as-is; this contradicts the earlier "no per-item tool logos" note in
  `graphical_abstract_asset_generation.md`, which is intentionally superseded by this document).
- Third-party icon licenses/sources should be documented in
  `graphical_abstract_asset_generation.md` (already done for the Python/NumPy additions).

## Content / real figures

- Every plot thumbnail is a **real, regenerated** figure via
  `paper/figures/generate_graphical_abstract_assets.py`, from one consistent running example
  (the paper's own Liquid Flow 2D quickstart case) — never a placeholder or an unrelated
  reused figure.
- Every plot is stripped of legend, colorbar, and axis/ticks (`_strip_legends()` helper) — even
  the log-log convergence plots are rendered axis-free for visual consistency with the rest of
  the page.
- **Meshes**: only the 3D prism mesh thumbnail (the 2D mesh thumbnail was removed as redundant).
- **Model**: boundary-conditions preview is a generated plot (domain colored by material zone +
  two colored boundary edges with a white halo for contrast), not the old legacy
  `figure1_boundary_conditions.png` (which carried a legend/colorbar/axes).
- **Post-processing**: spatial pressure contourf + 4-probe temporal line plot; log-convergence
  heatmap + computational-metrics (assembly/linear-solver time) line plot.
- **Multiple simulations → Compare / Studies**: the real rendered plots (MeshSeries difference
  field, absolute-error convergence plot) were tried but ultimately judged **not legible at this
  display size** and removed — these cards use the abstract box/icon representation only (no
  embedded plot image).
  - Note for future reference: the Studies card's underlying data generator
    (`generate_convergence_study()` in the asset script) computes the **absolute error
    (maximum)** convergence metric using `VolumetricFlowRate` as the variable (not `pressure`,
    whose domain max/min are prescribed Dirichlet values and therefore trivially zero error) —
    keep this in mind if the plot is ever reinstated.
- **Multiple simulations → Chain**: purely symbolic (3 boxes + arrows), no real figure — this
  card represents restart/chained-simulation workflows conceptually only.

## Fixes applied post-v43 (continuing from a different session)

- **Meshes**: the 3D prism thumbnail moved to **Model** (captioned "BHE mesh"); Meshes now shows
  two new real generated images instead — "Extracted boundaries" (a 14-material layered mesh,
  regenerated from `examples.load_meshseries_THM_2D_PVD()` + `Meshes.from_mesh()`, same content
  as `docs/examples/howto_preprocessing/plot_extract_boundaries.py`'s first figure) and "PyVista
  surfaces" (a 3D tetrahedral mesh from Gaussian pyvista surfaces, same content as
  `docs/examples/howto_preprocessing/plot_meshlib_pyvista_input.py`, requires `tetgen` on
  `PATH`). Both stripped of axis/legend, matching the rest of the page.
- **Model**: now shows two images — "BHE mesh" (the relocated 3D prism thumbnail) and "Domain &
  boundaries" (a new generated image: the Selke Basin example, `examples.load_meshes_selke()`,
  plotted with `show_edges=False, cbar=False`, legend/axis stripped, well markers removed).
- **Workflow orchestration**'s fork/join diagram: the 5 boxes were bespoke SVG `<rect>`s
  (`.flow-box-blue`/`.flow-box-red`, sized 40-44px) that could silently drift out of sync with
  the `.sim-box` class used everywhere else for "= SINGLE SIMULATION" (60x44px). Rebuilt as real
  `.sim-box`/`.sim-box.flow-red` `<div>`s, absolutely positioned inside a `.flow-diagram-wrap`
  container (same pattern as `.conv-stack`'s stacked boxes) — size/radius/shadow can now never
  drift from the canonical box, only position and color are overridden per-instance. Only the
  connecting arrows remain as an SVG overlay.
- **`render_graphical_abstract_svg()` (in `generate_graphical_abstract_assets.py`) had a real
  bug**: the exported `paper/figures/graphical_abstract.svg` (what `paper.md` actually embeds)
  was silently missing the entire "Application" section and the bottom compatibility/notes line.
  Root cause: the print-to-PDF step's page-height margin (a modest constant, computed from an
  on-screen content-height measurement) was nowhere near enough — Chromium's print pipeline
  needs roughly 1.7x the on-screen content height before it stops silently overflowing onto a
  second, dropped PDF page (confirmed by bisecting page height while checking `pdfinfo`'s page
  count; root cause not otherwise identified — `.page`'s `overflow-x: auto` was a plausible
  culprit but disabling it alone didn't fix the pagination). Fixed by rendering on a deliberately
  oversized fixed page (3200×6000px), asserting via `pdfinfo` that this always produces exactly 1
  PDF page (raises loudly instead of silently truncating if the page ever grows past this
  margin again), then measuring the *true* content bounding box from the rendered PDF itself and
  cropping the final SVG's `width`/`height`/`viewBox` down to that — instead of trusting a
  separate screenshot-based pre-measurement to predict print-layout size.

## Fixes applied post-v43, round 2

- **MeshSeries pictures**: "Spatial plot" renamed to "Aggregation" (the image itself, the Elder
  benchmark max-saturation contourf from `docs/examples/howto_postprocessing/plot_aggregate.py`,
  was already swapped in an earlier round). "Temporal plot" replaced with "Observation points": a
  new generated image (`generate_meshseries_observation_points()`), the probe-line plot from
  `docs/examples/howto_plot/plot_observation_points.py` (`mesh_series.probe()` at two rows of 4
  points, `plot_line(..., labels=..., monospace=True)`), same Elder-benchmark mesh series as the
  picture next to it. The old liquid-flow-based `generate_meshseries_spatial()` /
  `generate_meshseries_temporal()` (and their now-unused `PROBE_POINTS`/`PROBE_LABELS` constants)
  were removed as dead code once both pictures switched to the Elder benchmark.
- **New "Plot" group in Post-processing**: 4 generated images from
  `docs/examples/howto_quickstart/plot_solid_mechanics.py` (`examples.load_mesh_mechanics_2D()`),
  chosen to show breadth across the variable/plot API: "Displacement" (vector field contourf),
  "Von Mises stress" (derived scalar from the stress tensor), "Principal stresses" (eigenvalue
  contourf with an eigenvector quiver overlay), "Dilatancy criterion" (an integrity-criterion
  derived variable). Laid out as two stacked `.thumb-pair` rows inside one `.thumb-cell`-styled
  group, same visual pattern as MeshSeries/Log.
- **Thumb-cell border seam removed**: `.thumb-pair img` no longer carries its own
  `border: 1px solid var(--border)` — previously this, combined with the newer `.thumb-cell`
  outer box (added in round 1), produced a visible double-border/seam directly between an image
  and its caption. The outer `.thumb-cell` border is now the only border framing image + caption
  together.
- **Thumb-cell height mismatch fixed**: `.thumb-pair` switched from `align-items: flex-start` to
  `align-items: stretch`, so the two `.thumb-cell` boxes in a row always match height — previously
  a 2-line caption (e.g. "Numerical convergence") made its box taller than a 1-line-caption
  neighbor (e.g. "Computational metrics"), leaving their bottom borders visibly misaligned.
- **Pint added to the compatibility/notes row**, right after NumPy: icon + "Pint" text, same
  pattern as the other library notes. The icon is the real official Pint logo (a pint-of-beer
  glass photo, `docs/_static/logo-full.jpg` in `hgrecco/pint` on GitHub — genuinely their brand
  mark, not a stand-in), whitespace-trimmed and saved as `assets/pint_logo.png`.
- **`--blue` brand color corrected**: sampled directly from the real OpenGeoSys wordmark logo
  (`assets/ogs_logo.png`, the "GeoSys" glyphs) via pixel-color counting → `#234DAC`, replacing the
  previous `#104EB2` which was darker/more saturated than the actual OpenGeoSys brand blue. This
  is the single `--blue` CSS variable used everywhere (OGSTools wordmark, headers, borders,
  `.sim-box`), so the fix applies page-wide, not just to the wordmark.

## Fixes applied post-v43, round 3

- **"Plot" group trimmed from 4 to 2 pictures**: "Displacement" and "Principal stresses" removed,
  keeping only "Von Mises stress" and "Dilatancy criterion" as a single `.thumb-pair` row (same
  layout as MeshSeries/Log), instead of two stacked rows. `generate_solidmech_displacement()` and
  `generate_solidmech_principal_stress()` (and their now-unused assets) removed from
  `generate_graphical_abstract_assets.py` as dead code.
- **Real screenshot embedded under Monitor**: the generic monitor SVG icon replaced with the
  actual "Bokeh log plot" screenshot from
  `docs/examples/howto_simulation/plot_010_simulate.html`'s live docs build
  (`_images/bokeh_logs.png`, saved as `assets/monitor_bokeh_log.png` — a real screenshot of
  `sim.log`'s bokeh-based live monitor, not something regenerable via the local script since it's
  captured from an interactive widget during Sphinx-Gallery's doc build). Captioned "Watch
  simulation progress with bokeh"; the existing "Live simulation progress" feature-description
  line stays below it.
- **MeshSeries feature-description reworded**: "Aggregation · Probing · Differencing" →
  "Spatial / Temporal plots · Variables with physical units" — ties the line more directly to the
  two pictures next to it (spatial aggregation contourf, temporal observation-points line plot)
  and calls out Pint-backed unit awareness (see the Pint compatibility-note addition above).
- **Column-height balancing (Pre-processing / Simulation Execution / Post-processing)**: with the
  new Monitor picture and the trimmed Plot section, Post-processing was the tallest column and the
  other two had grown noticeably shorter, leaving uneven trailing white space beneath the
  `.box col` borders (which are stretched to match the tallest column via `align-items: stretch`
  on `.row`). Narrowed the gap (not eliminated - the columns' *content* naturally differs in
  length) by enlarging two image heights that had headroom to grow without looking oversized: the
  new Monitor bokeh screenshot (170px → 280px, also improves legibility of its small subplot
  labels) and Model's two pictures (126px → 170px, via inline `style` overrides on just those
  `<img>` tags rather than changing the shared `.thumb-pair img` rule used everywhere else on the
  page).

## Fixes applied post-v43, round 4

- **"BHE mesh" renamed to "Borehole heat exchanger"** (Model group) — spells out the acronym for
  readers unfamiliar with it.
- **Model's two pictures swapped left/right**: "Domain & boundaries" now comes first, "Borehole
  heat exchanger" second (previously the reverse).
- **Caption swap reverted/corrected**: round 3 had moved "Spatial / Temporal plots · Variables
  with physical units" onto MeshSeries's feature-description line, but that caption was meant for
  **Plot**, not MeshSeries. MeshSeries's feature-description line is restored to its original
  wording, "Aggregation · Probing · Differencing"; Plot's feature-description line (previously
  "contourf · Variable presets (stress, strain, displacement)") now reads "Spatial / Temporal
  plots · Variables with physical units".
