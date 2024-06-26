from string import ascii_uppercase
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ogstools.meshlib.data_processing import sample_polyline
from ogstools.plot import contourf, utils
from ogstools.plot.shared import spatial_quantity
from ogstools.propertylib.properties import Property, get_preset


def linesample(
    mesh: pv.UnstructuredGrid,
    x: str,
    y_property: str | Property,
    profile_points: np.ndarray,
    ax: plt.Axes,
    resolution: int | None = 100,
    **kwargs: Any,
) -> plt.Axes:
    """
    Plot selected properties obtained from sample_over_polyline function,
    this function calls to it internally. Values provided in param x and y
    refer to columns of the DataFrame returned by it.

    :param x: Value to be used on x-axis of the plot
    :param y_property: Values to be used on y-axis of the plot
    :param profile_points: Points defining the profile (and its segments)
    :param ax: User-created array of Matplotlib axis object
    :param resolution: Resolution of the sampled profile. Total number of
        points within all profile segments.
    :param resolution: Resolution of the sampled profile. Total number of
        points within all profile segments.
    :param kwargs: Optional keyword arguments passed to matplotlib.pyplot.plot
        to customize plot options like a line label (for auto legends), linewidth,
        antialiasing, marker face color.

    :return: Matplotlib Axes object
    """

    mesh_property = get_preset(y_property, mesh)
    mesh_sp, _ = sample_polyline(
        mesh, mesh_property, profile_points, resolution
    )

    assert isinstance(ax, plt.Axes)

    spatial_qty = spatial_quantity(mesh)
    kwargs.setdefault("label", mesh_property.data_name)
    kwargs.setdefault("color", mesh_property.color)
    kwargs.setdefault("linestyle", mesh_property.linestyle)
    if "ls" in kwargs:
        kwargs.pop("linestyle")

    utils.update_font_sizes(axes=ax, fontsize=kwargs.pop("fontsize", 20))
    ax.plot(
        spatial_qty.transform(mesh_sp[x]),
        mesh_sp[mesh_property.data_name],
        **kwargs,
    )
    ax.set_xlabel("Profile distance / " + spatial_qty.output_unit)
    ax.set_ylabel(mesh_property.get_label())

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
            y_property=property_current,
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
    utils.update_font_sizes(fig=fig)
    fig.tight_layout()

    return fig, ax
