# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Specialized plot features."""

from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import patheffects
from matplotlib.collections import PolyCollection
from matplotlib.transforms import blended_transform_factory as btf

from ogstools.propertylib import Vector

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
            x_b, y_b = setup.length.transform(
                edges.points[edges.lines % edges.n_points].T[[x_id, y_id]]
            )
            lw = setup.rcParams_scaled["lines.linewidth"]
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
        verts = setup.length.transform(np.delete(cell_pts, projection, -1))
        lw = 0.5 * setup.rcParams_scaled["lines.linewidth"]
        pc = PolyCollection(
            verts, fc="None", ec="black", lw=lw  # type: ignore[arg-type]
        )
        ax.add_collection(pc)


def plot_streamlines(
    ax: plt.Axes,
    mesh: pv.DataSet,
    mesh_property: Vector,
    projection: Optional[int] = None,
    plot_type: Literal["streamlines", "arrows", "lines"] = "streamlines",
) -> None:
    """
    Plot the vector streamlines or arrows on a matplotlib axis.

    :param ax:              Matplotlib axis to plot onto
    :param mesh:            Mesh containing the vector property
    :param mesh_property:   Vector property to visualize
    :param projection:      Index of flat dimension (e.g. 2 for z axis),
                            gets automatically determined if not given
    :param plot_type:       Whether to plot streamlines, arrows or lines.
    """
    if (n_pts := setup.num_streamline_interp_pts) is None:
        return
    if plot_type != "streamlines":
        n_pts = 50
    if projection is None:
        mean_normal = np.abs(
            np.mean(mesh.extract_surface().cell_normals, axis=0)
        )
        projection = int(np.argmax(mean_normal))
    x_id, y_id = np.delete([0, 1, 2], projection)
    bounds = [float(b) for b in mesh.bounds]
    x = np.linspace(bounds[2 * x_id], bounds[2 * x_id + 1], n_pts)
    y = np.linspace(bounds[2 * y_id], bounds[2 * y_id + 1], n_pts)
    z = np.array([np.mean(mesh.points[..., projection])])

    _mesh = mesh.copy()
    for key in _mesh.point_data:
        if key not in [mesh_property.data_name, mesh_property.mask]:
            del _mesh.point_data[key]
    grid = pv.RectilinearGrid(
        [x, y, z][x_id], [x, y, z][y_id], [x, y, z][projection]
    )
    grid = grid.sample(_mesh, pass_cell_data=False)
    values = mesh_property.transform(grid.point_data[mesh_property.data_name])
    values[np.argwhere(grid["vtkValidPointMask"] == 0), :] = np.nan
    if np.shape(values)[-1] == 3:
        values = np.delete(values, projection, 1)
    val = np.reshape(values, (n_pts, n_pts, 2))

    if mesh_property.mask in grid.point_data:
        mask = np.reshape(grid.point_data[mesh_property.mask], (n_pts, n_pts))
        val[mask == 0, :] = 0
    val_norm = np.linalg.norm(np.nan_to_num(val), axis=-1)
    lw = 2.5 * val_norm / max(1e-16, np.max(val_norm))
    lw *= setup.rcParams_scaled["lines.linewidth"]
    x_g, y_g = setup.length.transform(np.meshgrid(x, y))
    if plot_type == "streamlines":
        ax.streamplot(x_g, y_g, val[..., 0], val[..., 1],
                      color="k", linewidth=lw, density=1.5)  # fmt: skip
    else:
        line_args = (
            dict(  # noqa: C408
                headlength=0, headaxislength=0, headwidth=1, pivot="mid"
            )
            if plot_type == "lines"
            else {}
        )
        scale = 1.0 / 0.03
        ax.quiver(x_g, y_g, val[..., 0], val[..., 1], **line_args, scale=scale)


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
    ax.set_ylim(top=float(setup.length.transform(np.max(contour_vals))))
    ax.fill_between(
        setup.length.transform(x_vals),
        setup.length.transform(y_vals),
        setup.length.transform(contour_vals),
        facecolor="lightgrey",
    )


def plot_contour(
    ax: plt.Axes, mesh: pv.DataSet, style: str, lw: int, projection: int = 2
) -> None:
    contour = mesh.extract_surface().strip(join=True)
    x_id, y_id = np.delete([0, 1, 2], projection)
    x, y = 1e-3 * contour.points[contour.lines[1:]].T[[x_id, y_id]]
    ax.plot(x, y, style, lw=lw)
