# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Specialized plot features."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import patheffects
from matplotlib.collections import PolyCollection
from matplotlib.transforms import blended_transform_factory as btf

from .shared import setup, spatial_quantity


def layer_boundaries(
    ax: plt.Axes, mesh: pv.UnstructuredGrid, projection: int
) -> None:
    """Plot the material boundaries of a surface on a matplotlib axis."""
    mat_ids = np.unique(mesh.cell_data["MaterialIDs"])
    x_id, y_id = np.delete([0, 1, 2], projection)
    for mat_id in mat_ids:
        m_i = mesh.threshold((mat_id, mat_id), "MaterialIDs")
        # the pyvista connectivity call adds RegionID cell data
        segments = m_i.extract_feature_edges().connectivity(largest=False)
        for reg_id in np.unique(segments.cell_data["RegionId"]):
            segment = segments.threshold((reg_id, reg_id), "RegionId")
            edges = segment.extract_surface().strip(True, 10000)
            x_b, y_b = spatial_quantity(mesh).transform(
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
            outline_eff = [patheffects.withStroke(linewidth=1, foreground="k")]
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
                path_effects=outline_eff,
            )


def element_edges(
    ax: plt.Axes, mesh: pv.UnstructuredGrid, projection: int
) -> None:
    """Plot the element edges of a surface on a matplotlib axis."""
    lin_mesh = mesh.linear_copy()
    cell_points = [lin_mesh.get_cell(i).points for i in range(lin_mesh.n_cells)]
    cell_types = [lin_mesh.get_cell(i).type for i in range(lin_mesh.n_cells)]
    for cell_type in np.unique(cell_types):
        cell_pts = [
            cp
            for cp, ct in zip(cell_points, cell_types, strict=False)
            if ct == cell_type
        ]
        verts = spatial_quantity(lin_mesh).transform(
            np.delete(cell_pts, projection, -1)
        )
        lw = 0.5 * setup.rcParams_scaled["lines.linewidth"]
        pc = PolyCollection(verts, fc="None", ec="black", lw=lw)
        ax.add_collection(pc)


def shape_on_top(
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
    contour_vals = [
        y + scaling * contour(x) for y, x in zip(y_vals, x_vals, strict=False)
    ]
    spatial = spatial_quantity(surf).transform
    ax.set_ylim(top=float(spatial(np.max(contour_vals))))
    ax.fill_between(
        spatial(x_vals),
        spatial(y_vals),
        spatial(contour_vals),
        facecolor="lightgrey",
    )


def outline(
    ax: plt.Axes, mesh: pv.DataSet, style: str, lw: int, projection: int = 2
) -> None:
    "Plot the outline of a mesh on a matplotlib ax object."
    contour = mesh.extract_surface().strip(join=True)
    x_id, y_id = np.delete([0, 1, 2], projection)
    x, y = spatial_quantity(mesh).transform(
        contour.points[contour.lines[1:]].T[[x_id, y_id]]
    )
    ax.plot(x, y, style, lw=lw)
