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
from matplotlib.collections import PolyCollection

from .shared import setup


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
            x_b, y_b = edges.points[edges.lines % edges.n_points].T[
                [x_id, y_id]
            ]
            ax.plot(x_b, y_b, "-k", lw=setup.linewidth)


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
        verts = np.delete(cell_pts, projection, -1)
        lw = 0.5 * setup.linewidth
        pc = PolyCollection(verts.tolist(), fc="None", ec="black", lw=lw)
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
    ax.set_ylim(top=float(np.max(contour_vals)))
    ax.fill_between(x_vals, y_vals, contour_vals, facecolor="lightgrey")
