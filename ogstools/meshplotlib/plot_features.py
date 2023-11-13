"""Specialized plot features."""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import patheffects
from matplotlib.collections import PolyCollection
from matplotlib.transforms import blended_transform_factory as btf

from ogstools.propertylib.property import Vector

from . import setup


def plot_layer_boundaries(
    ax: plt.Axes, surf: pv.DataSet, projection: int
) -> None:
    """Plot the material boundaries of a surface on a matplotlib axis."""
    mat_ids = np.unique(surf.cell_data["MaterialIDs"])
    x_id, y_id = np.delete([0, 1, 2], projection)
    for mat_id in mat_ids:
        m_i = surf.threshold((mat_id, mat_id), "MaterialIDs")
        # the pyvista connectivity call adds RegionID cell data
        segments = m_i.extract_feature_edges().connectivity(largest=False)
        for reg_id in np.unique(segments.cell_data["RegionId"]):
            segment = segments.threshold((reg_id, reg_id), "RegionId")
            edges = segment.extract_surface().strip(True, 10000)
            x_b, y_b = setup.length.strip_units(
                edges.points[edges.lines % edges.n_points].T[[x_id, y_id]]
            )
            lw = 0.5 * setup.rcParams_scaled["lines.linewidth"]
            ax.plot(x_b, y_b, "-k", lw=lw)
        x_pos = 0.01 if mat_id % 2 == 0 else 0.99
        ha = "left" if mat_id % 2 == 0 else "right"
        x_b_lim = x_b.min() if mat_id % 2 == 0 else x_b.max()
        y_pos = np.mean(y_b[x_b == x_b_lim])
        if not setup.embedded_region_names_color:
            continue
        if setup.material_names is not None and mat_id in setup.material_names:
            c = setup.embedded_region_names_color
            m = ">" if mat_id % 2 == 0 else "<"
            outline = [patheffects.withStroke(linewidth=1, foreground="k")]
            plt.scatter(
                round(x_pos) + 0.2 * (x_pos - round(x_pos)),
                y_pos,
                transform=btf(ax.transAxes, ax.transData),
                color=c,
                marker=m,
            )
            plt.text(
                x_pos,
                y_pos,
                setup.material_names[mat_id],
                fontsize=plt.rcParams["font.size"] * 0.75,
                transform=btf(ax.transAxes, ax.transData),
                color=c,
                weight="bold",
                ha=ha,
                va="center",
                path_effects=outline,
            )


def plot_element_edges(ax: plt.Axes, surf: pv.DataSet, projection: int) -> None:
    """Plot the element edges of a surface on a matplotlib axis."""
    cell_points = [surf.get_cell(i).points for i in range(surf.n_cells)]
    cell_types = [surf.get_cell(i).type for i in range(surf.n_cells)]
    for cell_type in np.unique(cell_types):
        cell_pts = [
            cp for cp, ct in zip(cell_points, cell_types) if ct == cell_type
        ]
        verts = setup.length.strip_units(np.delete(cell_pts, projection, -1))
        lw = 0.5 * setup.rcParams_scaled["lines.linewidth"]
        pc = PolyCollection(verts, fc="None", ec="black", lw=lw)
        ax.add_collection(pc)


def plot_streamlines(
    ax: plt.Axes, surf: pv.DataSet, property: Vector, projection: int
) -> None:
    """Plot vector streamlines on a matplotlib axis."""
    if (n_pts := setup.num_streamline_interp_pts) is None:
        return
    x_id, y_id = np.delete([0, 1, 2], projection)
    bounds = [float(b) for b in surf.bounds]
    x = np.linspace(bounds[2 * x_id], bounds[2 * x_id + 1], n_pts)
    y = np.linspace(bounds[2 * y_id], bounds[2 * y_id + 1], n_pts)
    z = np.array([np.mean(surf.points[..., projection])])

    _surf = surf
    for key in _surf.point_data:
        if key not in [property.data_name, property.mask]:
            del _surf.point_data[key]
    grid = pv.RectilinearGrid(
        [x, y, z][x_id], [x, y, z][y_id], [x, y, z][projection]
    )
    grid = grid.sample(_surf, pass_cell_data=False)
    if np.shape(grid.point_data[property.data_name])[-1] == 3:
        grid.point_data[property.data_name] = np.delete(
            grid.point_data[property.data_name], projection, 1
        )
    val = np.reshape(
        property.strip_units(grid.point_data[property.data_name]),
        (n_pts, n_pts, 2),
    )

    if property.mask in grid.point_data:
        mask = np.reshape(grid.point_data[property.mask], (n_pts, n_pts))
        val[mask == 0, :] = 0
    val_norm = np.linalg.norm(val, axis=-1)
    lw = 2.5 * val_norm / max(1e-16, np.max(val_norm))
    lw *= setup.rcParams_scaled["lines.linewidth"]

    x_g, y_g = setup.length.strip_units(np.meshgrid(x, y))
    ax.streamplot(x_g, y_g, val[..., 0], val[..., 1], color="k", linewidth=lw)


def plot_on_top(
    ax: plt.Axes,
    surf: pv.DataSet,
    contour: Callable[[np.ndarray], np.ndarray],
    scaling: float = 1.0,
) -> None:
    normal = np.abs(np.mean(surf.extract_surface().cell_normals, axis=0))
    projection = int(np.argmax(normal))

    XYZ = surf.extract_feature_edges().points
    df_pts = pd.DataFrame(
        np.delete(XYZ, projection, axis=1), columns=["x", "y"]
    )
    x_vals = df_pts.groupby("x")["x"].agg(np.mean).to_numpy()
    y_vals = df_pts.groupby("x")["y"].agg(np.max).to_numpy()
    contour_vals = [y + scaling * contour(x) for y, x in zip(y_vals, x_vals)]
    ax.set_ylim(top=setup.length.strip_units(np.max(contour_vals)))
    ax.fill_between(
        setup.length.strip_units(x_vals),
        setup.length.strip_units(y_vals),
        setup.length.strip_units(contour_vals),
        facecolor="lightgrey",
    )


def plot_contour(
    ax: plt.Axes, mesh: pv.DataSet, style: str, lw: int, projection: int = 2
):
    contour = mesh.extract_surface().strip(join=True)
    x_id, y_id = np.delete([0, 1, 2], projection)
    x, y = 1e-3 * contour.points[contour.lines[1:]].T[[x_id, y_id]]
    ax.plot(x, y, style, lw=lw)
