from string import ascii_uppercase
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ogstools.meshlib.data_processing import sample_polyline
from ogstools.plot import contourf
from ogstools.plot.shared import spatial_quantity
from ogstools.plot.utils import update_font_sizes
from ogstools.propertylib.properties import Property, get_preset


def _color_twin_axes(axes: list[plt.Axes], colors: list) -> None:
    for ax_temp, color_temp in zip(axes, colors, strict=False):
        ax_temp.tick_params(axis="y", which="both", colors=color_temp)
        ax_temp.yaxis.label.set_color(color_temp)
    # Axis spine color has to be applied on twin axis for both sides
    axes[1].spines["left"].set_color(colors[0])
    axes[1].spines["right"].set_color(colors[1])


def linesample(
    mesh: pv.UnstructuredGrid,
    x: str,
    y: str | Property | list[str] | list[Property] | np.ndarray,
    profile_points: np.ndarray,
    ax: plt.Axes | np.ndarray | None = None,
    fontsize: int = 20,
    twinx: bool | None = False,
    resolution: int | None = 100,
) -> plt.Axes:
    """
    Plot selected properties obtained from sample_over_polyline function,
    this function calls to it internally. Values provided in param x and y
    refer to columns of the DataFrame returned by it.

    :param x: Value to be used on x-axis of the plot
    :param y: Values to be used on y-axis of the plot
    :param profile_points: Points defining the profile (and its segments)
    :param ax: User-created array of Matplotlib axis object
    :param resolution: Resolution of the sampled profile. Total number of
        points within all profile segments.
    :param fontsize: Font size to be used for all captions and labels in the
        plot
    :param twinx: Enable plotting second property on twin-x axis (only works
        if exactly two properties are provided in props param)
    :param resolution: Resolution of the sampled profile. Total number of
        points within all profile segments.

    :return: Matplotlib Axes object
    """
    # TODO: Vector properties with 2 values should be handled automatically
    if isinstance(y, list | np.ndarray) and twinx:
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

    _mesh_properties = [y] if not isinstance(y, list | np.ndarray) else y
    mesh_properties = [get_preset(y_i, mesh) for y_i in _mesh_properties]

    mesh_sp, _ = sample_polyline(
        mesh, mesh_properties, profile_points, resolution
    )

    assert isinstance(ax, plt.Axes)
    if twinx:
        ax_twinx = cast(plt.Axes, ax.twinx())

    spatial_qty = spatial_quantity(mesh)
    for prop in mesh_properties:
        ax.plot(
            spatial_qty.transform(mesh_sp[x]),
            mesh_sp[prop.data_name],
            label=prop.data_name,
            color=prop.color,
            linestyle=prop.linestyle,
        )
        # TODO: this shouldn't be hard-coded
        ax.set_xlabel("Profile distance / " + spatial_qty.output_unit)
        ax.set_ylabel(prop.get_label())
        # % TODO: rethink this awkward structure, maybe check if units match?
        if twinx:
            # Break after first property, as second one will be
            # handled outside of the loop
            break

    if twinx:
        ax_twinx.plot(
            spatial_qty.transform(mesh_sp[x]),
            mesh_sp[mesh_properties[-1].data_name],
            label=mesh_properties[-1].data_name,
            color=mesh_properties[-1].color,
            linestyle=mesh_properties[-1].linestyle,
        )
        ax_twinx.set_ylabel(mesh_properties[-1].get_label())
        ax_twinx.minorticks_on()
        _color_twin_axes(
            [ax, ax_twinx],
            [mesh_properties[0].color, mesh_properties[-1].color],
        )
        update_font_sizes(axes=ax_twinx, fontsize=fontsize)

    update_font_sizes(axes=ax, fontsize=fontsize)
    # TODO: this should be in apply_mpl_style()
    ax.grid(which="major", color="lightgrey", linestyle="-")
    ax.grid(which="minor", color="0.95", linestyle="--")
    ax.minorticks_on()

    return ax


def linesample_contourf(
    mesh: pv.UnstructuredGrid,
    properties: str | list | Property,
    profile_points: np.ndarray,
    profile_plane: tuple | list = (0, 1),
    resolution: int | None = None,
    plot_nodal_pts: bool | None = True,
    nodal_pts_labels: str | list | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Default plot for the data obtained from sampling along a profile on a mesh.

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
        2, len(properties), figsize=(len(properties) * 13, 12)
    )
    ax = ax.reshape((2, len(properties)))

    spatial_qty = spatial_quantity(mesh)
    for property_id, property_current in enumerate(properties):
        contourf(
            mesh,
            property_current,
            fig=fig,
            ax=ax[0, property_id],
        )
        linesample(
            mesh,
            x="dist",
            y=property_current,
            profile_points=profile_points,
            ax=ax[1, property_id],
            resolution=resolution,
        )

        if plot_nodal_pts:
            if nodal_pts_labels is None:
                nodal_pts_labels = list(
                    ascii_uppercase[0 : len(profile_points)]
                )
            ax[0][property_id].plot(
                spatial_qty.transform(profile_points[:, profile_plane[0]]),
                spatial_qty.transform(profile_points[:, profile_plane[1]]),  # type: ignore[index]
                "-*",
                linewidth=2,
                markersize=7,
                color="orange",
            )
            for nodal_pt_id, nodal_pt in enumerate(dist_at_knot):
                ax[0][property_id].text(
                    spatial_qty.transform(
                        profile_points[:, profile_plane[0]][nodal_pt_id]
                    ),  # type: ignore[index]
                    spatial_qty.transform(
                        profile_points[:, profile_plane[1]][nodal_pt_id]
                    ),  # type: ignore[index]
                    nodal_pts_labels[nodal_pt_id],
                    color="orange",
                    fontsize=15,
                    ha="left",
                    va="center",
                )
                ax[1][property_id].axvline(
                    spatial_qty.transform(nodal_pt),
                    linestyle="--",
                    color="orange",
                    linewidth=2,
                )
            ax_twiny = ax[1][property_id].twiny()
            ax_twiny.set_xlim(ax[1][property_id].get_xlim())
            ax_twiny.set_xticks(
                spatial_qty.transform(dist_at_knot),
                nodal_pts_labels,
                color="orange",
            )
    update_font_sizes(fig=fig)
    fig.tight_layout()

    return fig, ax
