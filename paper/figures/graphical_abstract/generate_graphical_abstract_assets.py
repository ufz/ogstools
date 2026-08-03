"""Generate the real image assets used by the OGSTools graphical abstract.

Each function is independently callable and writes one PNG into ``assets/``
(next to this script). Only generates the images actually embedded in
``graphical_abstract.html`` - see ``graphical_abstract_requirements.md`` in
this same folder for what's used where, and
``graphical_abstract_asset_generation.md`` for design rationale.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

import ogstools as ot
from ogstools.examples import load_project_simple_lf


def _strip_legends(fig: plt.Figure) -> None:
    """Remove any legend/colorbar/axis decoration from ``fig``."""
    for ax in list(fig.axes):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if (
            getattr(ax, "_colorbar", None) is not None
            or ax.get_label() == "<colorbar>"
        ):
            ax.remove()
            continue
        ax.set_axis_off()
    fig.tight_layout()


ASSETS_DIR = Path(__file__).parent / "assets"

import os

# Only needed the first time `assets/ogs_logo.png` is generated - that file
# is committed to the repo, so a plain checkout never needs this at all.
# Override with the OGS_LOGO_SOURCE env var if you do need to regenerate it
# and don't happen to have an OGS web checkout at this exact path.
OGS_LOGO_SOURCE = Path(
    os.environ.get(
        "OGS_LOGO_SOURCE", "~/o/wt/6.5.6/web/static/images/OGS-Logo.png"
    )
).expanduser()


def _liquid_flow_simulation(n_edge_cells: int = 8) -> ot.Simulation:
    """The one running example everything below is generated from.

    Same setup as the paper's own quickstart example
    (docs/examples/howto_quickstart/plot_framework.py): a 2D rectangular
    domain with a prescribed pressure difference between the left and right
    boundaries, run as a transient LIQUID_FLOW process (11 time steps).

    :param n_edge_cells: mesh resolution (cells per edge).
    """
    project = load_project_simple_lf()
    meshes = ot.Meshes.from_gmsh(
        ot.gmsh_tools.rect((8, 4), n_edge_cells, 2), log=False
    )
    meshes["left"].point_data["pressure"] = 2.9e7
    meshes["right"].point_data["pressure"] = 3.1e7
    model = ot.Model(project=project, meshes=meshes)
    return model.run()


def generate_mesh_3d_prism(out_name: str = "mesh_3d_prism.png") -> Path:
    """3D BHE (Borehole Heat Exchanger) mesh (Meshes).

    Reuses ogstools' own ``howto_preprocessing/plot_gen_bhe_mesh`` example -
    the "simple prism mesh" case and its ``load_and_plot`` - instead of a
    generic demo cuboid, so this thumbnail shows a real, documented OGSTools
    workflow (layered soil, groundwater layer, three BHEs, submesh
    wireframes) rather than an unrelated placeholder shape.
    """
    import pyvista as pv
    from shapely import Polygon

    from ogstools.meshes.gmsh_BHE import BHE, Groundwater, gen_bhe_mesh

    pv.OFF_SCREEN = True

    bhe_meshes = gen_bhe_mesh(
        model_area=Polygon.from_bounds(xmin=0, ymin=0, xmax=150, ymax=100),
        layer=[50, 50, 50],
        groundwater=Groundwater(
            begin=-30,
            isolation_layer_id=1,
            upstream=(179, 181),
            downstream=(359, 1),
        ),
        BHE_Array=[
            BHE(x=50, y=40, z_begin=-1, z_end=-60, borehole_radius=0.076),
            BHE(x=50, y=50, z_begin=-1, z_end=-60, borehole_radius=0.076),
            BHE(x=50, y=60, z_begin=-1, z_end=-60, borehole_radius=0.076),
        ],
        refinement_area=Polygon.from_bounds(xmin=40, ymin=30, xmax=60, ymax=70),
        meshing_type="prism",
        meshname="bhe_prism",
    )

    bhe_line = bhe_meshes.domain.extract_cells_by_type(pv.CellType.LINE)
    offsets = [(0, 0, 10), (0, 0, -10), (10, 0, 0), (-10, 0, 0)]
    plotter = ot.plot.contourf(
        bhe_meshes.domain.clip("x", origin=bhe_line.center, crinkle=True),
        ot.variables.material_id,
    )
    plotter.add_mesh(bhe_meshes.domain, style="wireframe", color="grey")
    plotter.add_mesh(bhe_line, color="r", line_width=3)
    for submesh, offset in zip(
        bhe_meshes.subdomains.values(), offsets, strict=True
    ):
        plotter.add_mesh(
            submesh.translate(offset), show_edges=True, color="lightgrey"
        )

    # strip scalar bar / orientation triad for visual consistency with the
    # rest of the graphical abstract's stripped-down assets
    plotter.remove_scalar_bar()
    plotter.hide_axes()

    plotter.off_screen = True
    plotter.window_size = (1400, 1000)
    out = ASSETS_DIR / out_name
    plotter.screenshot(out, transparent_background=True)
    return out


def generate_meshseries_spatial_aggregate(
    out_name: str = "meshseries_spatial_aggregate.png",
) -> Path:
    """Spatial contourf of max saturation over time (MeshSeries), used as the
    "Spatial plot" picture instead of :func:`generate_meshseries_spatial` -
    the first figure from
    docs/examples/howto_postprocessing/plot_aggregate.py (Elder benchmark,
    ``examples.load_meshseries_CT_2D_XDMF()``), not from the liquid-flow
    running example used everywhere else on the page. Note this breaks the
    4-marked-points correspondence with the "Temporal plot" picture next to
    it (a deliberate choice, not an oversight).
    """
    mesh_series = ot.examples.load_meshseries_CT_2D_XDMF().scale(time="a")
    saturation = ot.variables.saturation
    mesh = mesh_series.aggregate_temporal(saturation, np.max)
    fig = ot.plot.contourf(mesh, saturation)
    _strip_legends(fig)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_meshseries_observation_points(
    out_name: str = "meshseries_observation_points.png",
) -> Path:
    """Probe-line plot (MeshSeries), used as the "Temporal plot" picture -
    the observation-points example from
    docs/examples/howto_plot/plot_observation_points.py (Elder benchmark,
    ``examples.load_meshseries_CT_2D_XDMF()``), matching the aggregate
    picture next to it which uses the same benchmark.
    """
    mesh_series = ot.examples.load_meshseries_CT_2D_XDMF().scale(time="a")
    saturation = ot.variables.saturation
    rows = np.array([np.linspace([0, 0, z], [120, 0, z], 4) for z in [60, 40]])
    labels = [
        [f"{i}: x={pt[0]: >5} z={pt[2]}" for i, pt in enumerate(pts)]
        for pts in rows
    ]
    probes = [mesh_series.probe(pts) for pts in rows]
    fig = probes[0].plot_line(saturation, labels=labels[0], monospace=True)
    fig.tight_layout()
    _strip_legends(fig)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_log_convergence(out_name: str = "log_convergence.png") -> Path:
    """Convergence plot (Log), from the same simulation as the MeshSeries plots."""
    sim = _liquid_flow_simulation()
    fig = sim.log.plot_convergence()
    _strip_legends(fig)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_log_computational_metrics(
    out_name: str = "log_computational_metrics.png",
) -> Path:
    """Computational/performance metric plot (Log): assembly & linear solver time."""
    sim = _liquid_flow_simulation()
    ts = sim.log.time_step().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(
        ts["time_step"],
        ts["assembly_time"] * 1000,
        marker="o",
        label="assembly time",
    )
    ax.plot(
        ts["time_step"],
        ts["linear_solver_time"] * 1000,
        marker="o",
        label="linear solver time",
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("time / ms")
    ax.grid(which="major", color="lightgrey")
    fig.tight_layout()
    _strip_legends(fig)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _solid_mechanics_mesh() -> ot.Mesh:
    """Shared mesh for the solid-mechanics plots below (stress analysis of a
    hole in a plate, from docs/examples/howto_quickstart/plot_solid_mechanics.py).
    """
    return ot.examples.load_mesh_mechanics_2D()


def generate_solidmech_von_mises(
    out_name: str = "solidmech_von_mises.png",
) -> Path:
    """Von Mises stress contourf (Plot), from plot_solid_mechanics.py."""
    fig = ot.plot.contourf(
        _solid_mechanics_mesh(), ot.variables.stress.von_Mises
    )
    _strip_legends(fig)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_solidmech_dilatancy(
    out_name: str = "solidmech_dilatancy.png",
) -> Path:
    """Dilatancy (integrity) criterion contourf (Plot), from plot_solid_mechanics.py."""
    fig = ot.plot.contourf(
        _solid_mechanics_mesh(), ot.variables.dilatancy_critescu_tot
    )
    _strip_legends(fig)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_model_boundary_conditions(
    out_name: str = "model_boundary_conditions.png",
) -> Path:
    """Minimal boundary-conditions preview (Model), no legend/axis/colorbar.

    Domain colored by material zone, left/right pressure boundaries picked
    out as thick colored edges in the brand colors - conveys the same idea
    as the old ``figure1_boundary_conditions.png`` (which carried a busy
    legend + colorbar + axes) without any of that clutter.
    """
    meshes = ot.Meshes.from_gmsh(ot.gmsh_tools.rect((8, 4), 8, 2), log=False)
    fig = ot.plot.contourf(
        meshes.domain, "MaterialIDs", show_edges=True, cbar=False
    )
    ax = fig.axes[0]
    # White halo behind each boundary line so it reads clearly regardless of
    # which material color (light or dark blue) it happens to sit on top of.
    for x, color in [(0, "#00B8D9"), (8, "#B5650A")]:
        ax.plot(
            [x, x],
            [0, 4],
            color="white",
            linewidth=15,
            solid_capstyle="butt",
            zorder=4,
            clip_on=False,
        )
        ax.plot(
            [x, x],
            [0, 4],
            color=color,
            linewidth=9,
            solid_capstyle="butt",
            zorder=5,
            clip_on=False,
        )
    _strip_legends(fig)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def copy_ogs_logo(out_name: str = "ogs_logo.png") -> Path:
    """Copy the official OpenGeoSys logo into the repo's own assets.

    Avoids depending on an external OGS checkout path; OGSTools is the
    official OpenGeoSys companion library, so reusing the project logo here
    is in-scope. The result is committed to the repo, so on a plain checkout
    this is a no-op - only someone regenerating it from scratch needs
    ``OGS_LOGO_SOURCE`` to point at a real OGS web checkout.
    """
    out = ASSETS_DIR / out_name
    if out.exists():
        return out
    if not OGS_LOGO_SOURCE.exists():
        raise FileNotFoundError(
            f"{out} does not exist yet and OGS_LOGO_SOURCE "
            f"({OGS_LOGO_SOURCE}) was not found either. Set the "
            "OGS_LOGO_SOURCE env var to a local OGS web checkout's "
            "OGS-Logo.png, or copy assets/ogs_logo.png in from git."
        )
    shutil.copyfile(OGS_LOGO_SOURCE, out)
    return out


def generate_pint_logo(out_name: str = "pint_logo.png") -> Path:
    """Fetch the real official Pint logo (notes row, compatibility icons).

    Pint's own logo *is* a photo of a pint-of-beer glass (a pun on the unit) -
    ``docs/_static/logo-full.jpg`` in the ``hgrecco/pint`` GitHub repo, also
    what's shown on Pint's own documentation homepage - not a stand-in.
    Whitespace-trimmed for a tighter thumbnail. Result is committed to the
    repo, so on a plain checkout this is a no-op (like :func:`copy_ogs_logo`);
    only regenerating from scratch needs network access.
    """
    out = ASSETS_DIR / out_name
    if out.exists():
        return out
    url = (
        "https://raw.githubusercontent.com/hgrecco/pint/master/"
        "docs/_static/logo-full.jpg"
    )
    with urllib.request.urlopen(url) as response:
        raw = response.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    # source is a lossy .jpg with faint off-white noise right at the edges -
    # threshold the diff (scale by 2, drop anything <=10) so that noise
    # doesn't widen the crop box beyond the glass itself.
    diff = ImageChops.difference(image, background)
    diff = ImageChops.add(diff, diff, 2.0, -20)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.save(out)
    return out


def generate_monitor_bokeh_log(out_name: str = "monitor_bokeh_log.png") -> Path:
    """Fetch the real "Bokeh log plot" screenshot (Monitor).

    A genuine screenshot from OGSTools' own live docs build -
    ``docs/examples/howto_simulation/plot_010_simulate.html``'s
    ``_images/bokeh_logs.png`` - showing ``sim.log``'s bokeh-based live
    monitor. Not regenerable locally: it's captured from an interactive
    widget during Sphinx-Gallery's doc build, not a plain matplotlib figure.
    Result is committed to the repo, so on a plain checkout this is a no-op;
    only regenerating from scratch needs network access.
    """
    out = ASSETS_DIR / out_name
    if out.exists():
        return out
    url = "https://ogstools.opengeosys.org/stable/_images/bokeh_logs.png"
    with urllib.request.urlopen(url) as response:
        raw = response.read()
    out.write_bytes(raw)
    return out


def _measure_pdf_content_box(pdf_path: Path) -> tuple[float, float]:
    """True content width/height (PDF pt) of a single-page PDF's page 1,
    measured from the origin - via rasterizing it and finding the
    non-background bounding box. Used to crop
    :func:`render_graphical_abstract_svg`'s deliberately oversized export
    back down to just the real content.
    """
    dpi = 100
    png_prefix = Path(__file__).parent / "_ga_crop_measure"
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            str(dpi),
            str(pdf_path),
            str(png_prefix),
        ],
        check=True,
    )
    png_path = png_prefix.with_name(png_prefix.name + "-1.png")
    image = Image.open(png_path).convert("RGB")
    background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    bbox = ImageChops.difference(image, background).getbbox()
    png_path.unlink()
    if not bbox:
        return image.width * 72 / dpi, image.height * 72 / dpi
    # bbox[2]/[3] (not width/height): content starts essentially at the page
    # origin (just the print CSS's own small padding), so we crop from (0, 0)
    # down to where real content ends, keeping that padding as a visual
    # border rather than measuring width/height in isolation.
    return bbox[2] * 72 / dpi, bbox[3] * 72 / dpi


def render_graphical_abstract_svg(
    out_name: str = "graphical_abstract.svg",
) -> Path:
    """Render ``graphical_abstract.html`` to a real vector SVG for ``paper.md``.

    The page is a flexbox/CSS layout, not vector shapes to begin with, so a
    naive HTML->SVG conversion can't reproduce it - but headless Chromium's
    print pipeline *does* preserve real vector text/paths (unlike a
    ``--screenshot``, which rasterizes everything). Pipeline:

    1. Print to PDF on a deliberately oversized page (see ``page_width`` /
       ``page_height`` below) - large enough that content never silently
       overflows onto a dropped second PDF page. This isn't just a safety
       margin: measured empirically (bisecting page height while watching
       ``pdfinfo``'s page count), this page needs roughly 1.7x the height it
       actually occupies once rendered before Chromium's print pipeline
       stops silently paginating it - e.g. 2970x4075px still overflows to a
       second page, 2970x4300px is the first size that doesn't, even though
       the real rendered content (measured *inside* a successful render) is
       only ~2475px tall, matching the on-screen height almost exactly. Root
       cause not fully understood; ``.page``'s ``overflow-x: auto`` was a
       plausible suspect (a scroll container has no meaning on a printed
       page) but disabling it alone didn't fix the pagination either - it's
       disabled below anyway since it's the semantically correct thing to do
       for a print/export context regardless. A ``pdfinfo`` page-count check
       below guards against this margin becoming insufficient again as the
       page grows - if it fires, widen ``page_width``/``page_height``.
    2. Rasterize that one page and measure the *true* content bounding box
       (:func:`_measure_pdf_content_box`) - the oversized page from step 1
       leaves a lot of dead space below/right of the real content that a
       reader shouldn't see.
    3. Convert the PDF to SVG with ``pdftocairo`` (poppler) - real vector
       text/shapes, only the already-raster plot thumbnails stay raster (as
       they must) - then crop the SVG's own
       ``width``/``height``/``viewBox`` down to the box measured in step 2.

    Written into ``paper/figures/`` (one level up from this script), matching
    where ``paper.md``'s other figures live - not into this folder's own
    ``assets/``.
    """
    html_path = Path(__file__).parent / "graphical_abstract.html"
    html = html_path.read_text(encoding="utf-8")

    page_width, page_height = 3200, 6000

    # Override body's flex-centering/gray background (used for the on-screen
    # draft view) so the print render shows only the white .page box itself,
    # not the surrounding flex-container backdrop, and disable .page's
    # overflow-x:auto (meaningless - and per the docstring, a plausible
    # pagination culprit - on a printed page). The file's real <style> block
    # sits *after* </head>, inside <body> - so this override must go even
    # later, right before </body>, to win the cascade at equal specificity.
    margin = 60
    page_css = (
        f"<style>@page {{ size: {page_width}px {page_height}px; margin: 0; }} "
        f"html, body {{ margin: 0; padding: {margin}px; "
        f"display: block; background: var(--page); }} "
        f".page {{ overflow: visible !important; }}</style>"
    )
    print_html_path = Path(__file__).parent / "_ga_print.html"
    print_html_path.write_text(
        html.replace("</body>", page_css + "</body>"), encoding="utf-8"
    )

    pdf_path = Path(__file__).parent / "_ga_render.pdf"
    subprocess.run(
        [
            "chromium",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={pdf_path}",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            print_html_path.resolve().as_uri(),
        ],
        check=True,
    )
    print_html_path.unlink()

    pdfinfo = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    ).stdout
    page_count = int(pdfinfo.split("Pages:")[1].split()[0])
    if page_count != 1:
        pdf_path.unlink()
        msg = (
            f"graphical_abstract.html print-rendered to {page_count} PDF "
            f"pages instead of 1 (page size {page_width}x{page_height}px "
            "wasn't enough headroom - see render_graphical_abstract_svg()'s "
            "docstring). The page's content has likely grown; widen "
            "page_width/page_height rather than silently exporting a "
            "truncated SVG (pdftocairo -svg on a multi-page PDF converts "
            "the *last* page, usually blank, not the first)."
        )
        raise RuntimeError(msg)

    content_width_pt, content_height_pt = _measure_pdf_content_box(pdf_path)

    out = Path(__file__).parent.parent / out_name
    subprocess.run(
        ["pdftocairo", "-svg", "-f", "1", "-l", "1", str(pdf_path), str(out)],
        check=True,
    )
    pdf_path.unlink()

    # Crop the SVG's own viewport down to the real content box measured
    # above (plus a small margin) - the print page itself was deliberately
    # oversized (see docstring) and pdftocairo carries that size straight
    # into the SVG's width/height/viewBox otherwise.
    crop_margin_pt = 20
    crop_w = round(content_width_pt + crop_margin_pt, 2)
    crop_h = round(content_height_pt + crop_margin_pt, 2)
    svg = out.read_text(encoding="utf-8")
    svg, n_subs = re.subn(
        r'width="[\d.]+pt" height="[\d.]+pt" viewBox="0 0 [\d.]+ [\d.]+"',
        f'width="{crop_w}pt" height="{crop_h}pt" viewBox="0 0 {crop_w} {crop_h}"',
        svg,
        count=1,
    )
    if n_subs != 1:
        msg = (
            "Could not find the expected root <svg width=...pt height=...pt "
            "viewBox=...> attributes to crop - pdftocairo's SVG output "
            "format may have changed."
        )
        raise RuntimeError(msg)
    out.write_text(svg, encoding="utf-8")

    return out


if __name__ == "__main__":
    ASSETS_DIR.mkdir(exist_ok=True)
    for fn in [
        generate_mesh_3d_prism,
        generate_model_boundary_conditions,
        copy_ogs_logo,
        generate_pint_logo,
        generate_monitor_bokeh_log,
        generate_meshseries_spatial_aggregate,
        generate_meshseries_observation_points,
        generate_log_convergence,
        generate_log_computational_metrics,
        generate_solidmech_von_mises,
        generate_solidmech_dilatancy,
    ]:
        path = fn()
        print(f"wrote {path}")
    print(f"wrote {render_graphical_abstract_svg()}")
