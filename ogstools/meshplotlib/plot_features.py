"""Specialized plot features."""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import patheffects
from matplotlib.collections import PolyCollection
from matplotlib.transforms import blended_transform_factory as btf
from scipy.interpolate import griddata

from ogstools.propertylib.property import VectorProperty

from . import setup


def plot_layer_boundaries(
    ax: plt.Axes, surf: pv.DataSet, projection: int
) -> None:
    """Plot the material boundaries of a surface on a matplotlib axis."""
    mat_ids = np.unique(surf.cell_data["MaterialIDs"])
    x_id, y_id = np.delete([0, 1, 2], projection)
    for mat_id in mat_ids:
        m_i = surf.threshold((mat_id, mat_id), "MaterialIDs")
        # the pyvista connectivity call add RegionID cell data
        segments = m_i.extract_feature_edges().connectivity(largest=False)
        for reg_id in np.unique(segments.cell_data["RegionId"]):
            segment = segments.threshold((reg_id, reg_id), "RegionId")
            edges = segment.extract_surface().strip(True, 10000)
            x_b, y_b = setup.length.values(
                edges.points[edges.lines % edges.n_points].T[[x_id, y_id]]
            )
            ax.plot(
                x_b,
                y_b,
                "-k",
                lw=0.5 * setup.rcParams_scaled["lines.linewidth"],
            )
        x_pos = 0.01 if mat_id % 2 == 0 else 0.99
        ha = "left" if mat_id % 2 == 0 else "right"
        x_b_lim = x_b.min() if mat_id % 2 == 0 else x_b.max()
        y_pos = np.mean(y_b[x_b == x_b_lim])

        if setup.material_names is not None and mat_id in setup.material_names:
            outline = [patheffects.withStroke(linewidth=1, foreground="k")]
            plt.text(
                x_pos,
                y_pos,
                setup.material_names[mat_id],
                fontsize=plt.rcParams["font.size"] * 0.75,
                transform=btf(ax.transAxes, ax.transData),
                color="w",
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
        verts = setup.length.values(np.delete(cell_pts, projection, -1))
        pc = PolyCollection(
            verts,
            fc="None",
            ec="black",
            lw=0.5 * setup.rcParams_scaled["lines.linewidth"],
        )
        ax.add_collection(pc)


def plot_streamlines(
    ax: plt.Axes, surf: pv.DataSet, property: VectorProperty, projection: int
) -> None:
    """Plot vector streamlines on a matplotlib axis."""
    if not setup.num_streamline_interp_pts:
        return
    x_id, y_id = np.delete([0, 1, 2], projection)
    x, y, z = (
        np.linspace(
            float(surf.bounds[c_id * 2]),
            float(surf.bounds[c_id * 2 + 1]),
            setup.num_streamline_interp_pts,
        )
        for c_id in [x_id, y_id, projection]
    )
    interp = surf.interpolate(pv.StructuredGrid(x, y, z))
    p_field = property.values(interp.point_data[property.data_name])
    if property.mask in interp.point_data:
        mask = interp.point_data[property.mask]
    else:
        mask = np.ones(len(interp.points))
    x_grid, y_grid = np.meshgrid(x, y)
    val = p_field.T[[x_id, y_id]].T
    u_grid = griddata(
        (interp.points[:, x_id], interp.points[:, y_id]),
        val[:, 0],
        (x_grid, y_grid),
        method="linear",
    )
    v_grid = griddata(
        (interp.points[:, x_id], interp.points[:, y_id]),
        val[:, 1],
        (x_grid, y_grid),
        method="linear",
    )
    val_grid = griddata(
        (interp.points[:, x_id], interp.points[:, y_id]),
        np.linalg.norm(val, axis=1),
        (x_grid, y_grid),
        method="linear",
    )
    mask_grid = griddata(
        (interp.points[:, x_id], interp.points[:, y_id]),
        mask,
        (x_grid, y_grid),
        method="cubic",
    )
    # interpolation of masked cell_data to points is a bit tricky
    # cubic interpolation and a threshold of 0.4 gives good results
    u_grid[mask_grid < 0.4] = np.nan
    v_grid[mask_grid < 0.4] = np.nan
    lw = 2.5 * val_grid / np.max(np.linalg.norm(val, axis=1))
    ax.streamplot(
        setup.length.values(x_grid),
        setup.length.values(y_grid),
        u_grid,
        v_grid,
        color="k",
        linewidth=lw * setup.rcParams_scaled["lines.linewidth"],
    )


def get_aspect(ax: plt.Axes) -> float:
    """Return the aspect ratio of a matplotlib axis."""
    figW, figH = ax.get_figure().get_size_inches()
    _, _, w, h = ax.get_position().bounds
    disp_ratio = (figH * h) / (figW * w)
    data_ratio = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (
        ax.get_xlim()[1] - ax.get_xlim()[0]
    )
    return disp_ratio / data_ratio


# def plot_contour(
#
#     ax_id: int,
#     vtu_path: str,
#     origin: np.ndarray,
#     slice: bool,
#     style: str,
#     lw: int,
# ):
#     vtu = pv.XMLUnstructuredGridReader(vtu_path).read()
#     if dim == 2:
#         contour_vtu = vtu.extract_surface().strip(join=True)
#     else:
#         if slice:
#             vtu = vtu.slice(normal=ax_normals[ax_id], origin=origin)
#         else:
#             vtu = vtu.extract_feature_edges()
#         contour_vtu = vtu.strip(join=True)

#     x_id, y_id = np.delete([0, 1, 2], ax_normal_id[ax_id])
#     x, y = 1e-3 * contour_vtu.points[contour_vtu.lines[1:]].T[[x_id, y_id]]
#     fig.axes[ax_id].plot(x, y, style, lw=lw)
