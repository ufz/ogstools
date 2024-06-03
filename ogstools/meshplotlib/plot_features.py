# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""Specialized plot features."""

import string
from typing import Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import patheffects
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure as mfigure
from matplotlib.transforms import blended_transform_factory as btf

import ogstools.meshplotlib as mpl
from ogstools.meshlib import sample_polyline
from ogstools.propertylib import Property, Vector
from ogstools.propertylib.properties import get_preset

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


def _vectorfield(
    mesh: pv.DataSet,
    mesh_property: Vector,
    projection: Optional[int] = None,
) -> tuple[float, float, float, float, float]:
    """
    Compute necessary data for streamlines or quiverplots.

    :param mesh:            Mesh containing the vector property
    :param mesh_property:   Vector property to visualize
    :param projection:      Index of flat dimension (e.g. 2 for z axis),
                            gets automatically determined if not given
    """

    n_pts = setup.num_streamline_interp_pts
    n_pts = 50 if n_pts is None else n_pts
    mean_normal = np.abs(np.mean(mesh.extract_surface().cell_normals, axis=0))
    if projection is None:
        projection = int(np.argmax(mean_normal))
    i_id, j_id = np.delete([0, 1, 2], projection)
    _mesh = mesh.copy()
    for key in _mesh.point_data:
        if key not in [mesh_property.data_name, mesh_property.mask]:
            del _mesh.point_data[key]
    i_pts = np.linspace(mesh.bounds[2 * i_id], mesh.bounds[2 * i_id + 1], n_pts)
    j_pts = np.linspace(mesh.bounds[2 * j_id], mesh.bounds[2 * j_id + 1], n_pts)
    i_size = i_pts[-1] - i_pts[0]
    j_size = j_pts[-1] - j_pts[0]
    grid = pv.Plane(
        _mesh.center, mean_normal, i_size, j_size, n_pts - 1, n_pts - 1
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
    i_grid, j_grid = setup.length.transform(np.meshgrid(i_pts, j_pts))
    return (i_grid, j_grid, val[..., 0], val[..., 1], lw)


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

    if (setup.num_streamline_interp_pts) is None:
        return
    i_grid, j_grid, u, v, lw = _vectorfield(mesh, mesh_property, projection)
    if plot_type == "streamlines":
        ax.streamplot(
            i_grid, j_grid, u, v, color="k", linewidth=lw, density=1.5
        )
    else:
        line_args: dict = (
            {
                "headlength": 0,
                "headaxislength": 0,
                "headwidth": 1,
                "pivot": "mid",
            }
            if plot_type == "line"
            else {}
        )
        scale = 1.0 / 0.03
        ax.quiver(i_grid, j_grid, u, v, **line_args, scale=scale)


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


def plot_profile(
    mesh: pv.UnstructuredGrid,
    properties: Union[str, list, Property],
    profile_points: np.ndarray,
    profile_plane: Union[tuple, list] = (0, 1),
    resolution: Optional[int] = None,
    plot_nodal_pts: Optional[bool] = True,
    nodal_pts_labels: Optional[Union[str, list]] = None,
) -> tuple[mfigure, plt.Axes]:
    """
    Default plot for the data obtained from sampling along a profile on a mesh.

    :param mesh: Mesh providing the data
    :param props: Properties to be read from the mesh
    :param profile_points: Points defining the profile (and its segments)
    :param resolution: Resolution of the sampled profile. Total number of \
          points within all profile segments.
    :param plot_nodal_pts: Plot and annotate all nodal points in profile
    :param nodal_pts_labels: Labels for nodal points (only use if \
        plot_nodal_points is set to True)
    :param twinx: Enable plotting second property on twin-x axis (only works \
        if exactly two properties are provided in props param)
    :param profile_plane: Define which coordinates to use if profile plane \
        is different than XY: [0, 2] for XZ, [1, 2] for YZ...

    :return: Tuple containing Matplotlib Figure and Axis objects
    """
    # TODO: Add support for plotting only geometry at top subplot and
    # lineplot with twinx in the bottom one
    if not isinstance(properties, list):
        properties = [properties]

    _, dist_at_knot = sample_polyline(
        mesh, properties, profile_points, resolution=resolution
    )

    fig, ax = plt.subplots(
        2, len(properties), figsize=(len(properties) * 10, 10)
    )
    ax = ax.reshape((2, len(properties)))

    for property_id, property_current in enumerate(properties):
        fig = mpl.plot(
            mesh,
            property_current,
            fig=fig,
            ax=ax[0, property_id],
        )
        ax[1, property_id] = lineplot(
            x="dist",
            y=property_current,
            mesh=mesh,
            profile_points=profile_points,
            ax=ax[1, property_id],
            resolution=resolution,
        )

        if plot_nodal_pts:
            if nodal_pts_labels is None:
                nodal_pts_labels = list(
                    string.ascii_uppercase[0 : len(profile_points)]
                )
            ax[0, property_id].plot(
                profile_points[:, profile_plane[0]],
                profile_points[:, profile_plane[1]],  # type: ignore[index]
                "-*",
                linewidth=2,
                markersize=7,
                color="orange",
            )
            for nodal_pt_id, nodal_pt in enumerate(dist_at_knot):
                ax[0, property_id].text(
                    profile_points[:, profile_plane[0]][nodal_pt_id],  # type: ignore[index]
                    profile_points[:, profile_plane[1]][nodal_pt_id],  # type: ignore[index]
                    nodal_pts_labels[nodal_pt_id],
                    color="orange",
                    fontsize=15,
                    ha="left",
                    va="center",
                )
                ax[1, property_id].axvline(
                    nodal_pt, linestyle="--", color="orange", linewidth=2
                )
            ax_twiny = ax[1, property_id].twiny()
            ax_twiny.set_xlim(ax[1, property_id].get_xlim())
            ax_twiny.set_xticks(dist_at_knot, nodal_pts_labels, color="orange")

    return fig, ax


def lineplot(
    x: str,
    y: Union[str, Property, list, np.ndarray],
    mesh: pv.UnstructuredGrid,
    profile_points: np.ndarray,
    ax: plt.axes = None,
    fontsize: int = 20,
    twinx: Optional[bool] = False,
    resolution: Optional[int] = 100,
) -> plt.axes:
    """
    Plot selected properties obtained from sample_over_polyline function, \
    this function calls to it internally. Values provided in param x and y\
    refer to columns of the DataFrame returned by it.

    :param x: Value to be used on x-axis of the plot
    :param y: Values to be used on y-axis of the plot
    :param mesh: Mesh to be sampled
    :param profile_points: Points defining the profile (and its segments)
    :param ax: User-created array of Matplotlib axis object
    :param resolution: Resolution of the sampled profile. Total number of \
          points within all profile segments.
    :param fontsize: Font size to be used for all captions and labels in the \
        plot
    :param twinx: Enable plotting second property on twin-x axis (only works \
        if exactly two properties are provided in props param)
    :param resolution: Resolution of the sampled profile. Total number of \
          points within all profile segments.

    :return: Matplotlib Axes object
    """
    # TODO: Vector properties with 2 values should be handled automatically
    if isinstance(y, (list, np.ndarray)) and twinx:
        if len(y) == 1:
            twinx = False
        elif len(y) > 2:
            err_msg = "Only two properties are accepted for plot with twin \
                x-axis. If more are provided, I don't know how to split them!"
            raise ValueError(err_msg)
        if isinstance(ax, np.ndarray):
            err_msg = "If you want me to plot on twinx, I need to know on \
                which axis, so I will accept only plt.axes as ax parameter!"
            raise ValueError(err_msg)

    y = [y] if not isinstance(y, list) else y
    y = [get_preset(y_i, mesh) for y_i in y]

    mesh_sp, _ = sample_polyline(
        mesh, y, profile_nodes=profile_points, resolution=resolution
    )

    if twinx:
        ax_twinx = ax.twinx()

    for prop in y:
        ax.plot(
            mesh_sp[x],
            mesh_sp[prop.data_name],
            label=prop.data_name,
            color=prop.color,
            linestyle=prop.linestyle,
        )
        # TODO: this shouldn't be hard-coded
        ax.set_xlabel("Profile distance / m")
        ax.set_ylabel(prop.get_label())
        # % TODO: rethink this awkward structure, maybe check if units match?
        if twinx:
            # Break after first property, as second one will be
            # handled outside of the loop
            break

    if twinx:
        ax_twinx.plot(
            mesh_sp[x],
            mesh_sp[y[-1].data_name],
            label=y[-1].data_name,
            color=y[-1].color,
            linestyle=y[-1].linestyle,
        )
        ax_twinx.set_ylabel(y[-1].get_label())
        ax_twinx.minorticks_on()
        mpl.color_twin_axes([ax, ax_twinx], [y[0].color, y[-1].color])
        mpl.update_font_sizes(ax=ax_twinx, label_axes="none", fontsize=fontsize)

    # Apply default meshplotlib styling
    mpl.update_font_sizes(ax=ax, label_axes="none", fontsize=fontsize)
    # TODO: this should be in apply_mpl_style()
    ax.grid(which="major", color="lightgrey", linestyle="-")
    ax.grid(which="minor", color="0.95", linestyle="--")
    ax.minorticks_on()

    return ax
